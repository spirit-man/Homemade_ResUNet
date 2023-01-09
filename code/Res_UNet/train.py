import transforms as T
import torch.optim as opt
import time
import torch
import os
from torch import nn
from typing import Union, Sequence
from datetime import datetime


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, mean: tuple, std: tuple,
                 hflip_prob=0.5, vflip_prob=0.5):
        min_size = [int(i * 0.5) for i in base_size]
        max_size = [int(i * 1.2) for i in base_size]

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEvalorTest:
    def __init__(self, mean: tuple, std: tuple):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train: int, mean: tuple, std: tuple):
    base_size = [1080, 1920]
    crop_size = [540, 960]

    assert train == 0 or train == 1 or train == 2, "please pass 0(train) or 1(val) or 2(test)!"
    if train == 0:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEvalorTest(mean=mean, std=std)


def save_checkpoint(path: str,
                    epoch: int,
                    modules: Union[nn.Module, Sequence[nn.Module]],
                    optimizers: Union[opt.Optimizer, Sequence[opt.Optimizer]],
                    safe_replacement: bool = True):
    # This function can be called both as
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, my_module, my_opt)
    # or
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, [my_module1, my_module2], [my_opt1, my_opt2])
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, opt.Optimizer):
        optimizers = [optimizers]

    # Data dictionary to be saved
    data = {
        'epoch': epoch,
        # Current time (UNIX timestamp)
        'time': time.time(),
        # State dict for all the modules
        'modules': [m.state_dict() for m in modules],
        # State dict for all the optimizers
        'optimizers': [o.state_dict() for o in optimizers]
    }

    # Safe replacement of old checkpoint
    temp_file = None
    if os.path.exists(path) and safe_replacement:
        # There's an old checkpoint. Rename it!
        temp_file = path + '.old'
        os.rename(path, temp_file)

    # Save the new checkpoint
    with open(path, 'wb') as fp:
        torch.save(data, fp)
        # Flush and sync the FS
        fp.flush()
        os.fsync(fp.fileno())

    # Remove the old checkpoint
    if temp_file is not None:
        os.unlink(path + '.old')


def load_checkpoint(path: str,
                    default_epoch: int,
                    modules: Union[nn.Module, Sequence[nn.Module]],
                    optimizers: Union[opt.Optimizer, Sequence[opt.Optimizer]],
                    verbose: bool = True):
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, opt.Optimizer):
        optimizers = [optimizers]

    # If there's a checkpoint
    if os.path.exists(path):
        # Load data
        data = torch.load(path, map_location=next(modules[0].parameters()).device)

        # Inform the user that we are loading the checkpoint
        if verbose:
            print(f"Loaded checkpoint saved at {datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S')}. "
                  f"Resuming from epoch {data['epoch']}")

        # Load state for all the modules
        for i, m in enumerate(modules):
            modules[i].load_state_dict(data['modules'][i])

        # Load state for all the optimizers
        for i, o in enumerate(optimizers):
            optimizers[i].load_state_dict(data['optimizers'][i])

        # Next epoch
        return data['epoch'] + 1
    else:
        return default_epoch


def train(epoch, train_loader, model, criterion, optimizer, device, num_classes):
    model.train()
    accumulated_loss = 0.
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        target = nn.functional.one_hot(target, num_classes)
        target = target.permute(0, 3, 1, 2)
        # 2. Forward
        outputs = model(inputs)
        target = target.argmax(dim=1).long()
        loss = criterion(outputs, target)
        print("train: ", epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()
        # 5. Accumulate loss
        accumulated_loss += loss.item()
    return accumulated_loss
