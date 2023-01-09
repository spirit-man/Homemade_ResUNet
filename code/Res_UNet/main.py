import os.path
import torch
import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from BlueAlgaeDataset import *
from utils import mkdir
from train import get_transform, train, load_checkpoint, save_checkpoint
from Res_UNet import ResUNet
from eval import eval
from torch.optim.lr_scheduler import ReduceLROnPlateau


# use compute_mean_std to calculate
mean = (121.90957298, 130.36595824, 117.22788076)
std = (38.19792449, 35.57061731, 41.42008869)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch ResUNet training")

    parser.add_argument("--data-path", default=r"..\..\dataset", help="Dataset root")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=1e-1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.6, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default=False,
                        type=bool, help='resume from checkpoint')
    parser.add_argument('--save-best', default=True,
                        type=bool, help='only save best dice weights')

    args = parser.parse_args()

    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    batch_size = args.batch_size
    num_classes = args.num_classes

    results_path = r".\train_results"
    if not os.path.exists(results_path):
        mkdir(results_path)
    results_file = os.path.join(results_path, "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    save_weights_path = r".\save_weights"
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    checkpoint_path = r".\checkpoints"
    if not os.path.exists(checkpoint_path):
        mkdir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pth")

    train_dataset = BlueAlgaeDataset(filepath=args.data_path,
                                   train=0,
                                   transforms=get_transform(train=0, mean=mean, std=std))
    val_dataset = BlueAlgaeDataset(filepath=args.data_path,
                                   train=1,
                                   transforms=get_transform(train=1, mean=mean, std=std))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=False)

    model = ResUNet(in_channels=3, num_classes=num_classes)
    model.to(device)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    # optimizer, scheduler and criterion can use others
    optimizer = torch.optim.SGD(params_to_optimize,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    writer = SummaryWriter()

    if args.resume:
        first_epoch = load_checkpoint(checkpoint_path, 0, model, optimizer)
    else:
        first_epoch = 0

    # start training
    for epoch in range(first_epoch, args.epochs):
        # train
        accumulated_loss = train(epoch, train_loader, model, criterion, optimizer, device, num_classes)
        mean_loss = accumulated_loss / train_loader.__len__()

        # save model and checkpoint
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if args.save_best is True:
            save_weights = os.path.join(save_weights_path, "best_model.pth")
            torch.save(save_file, save_weights)
        else:
            save_weights = os.path.join(save_weights_path, "model_{}.pth".format(epoch))
            torch.save(save_file, save_weights)

        save_checkpoint(checkpoint_path, epoch, model, optimizer)

        # evaluation
        loss_val, accuracy, F1_score, MIoU = eval(epoch, val_loader, model, criterion, device, num_classes)

        # write into tensorboard
        writer.add_scalar('loss/train', mean_loss, global_step=epoch)
        writer.add_scalar('loss/val', loss_val, global_step=epoch)
        writer.add_scalar('accuracy/val', accuracy, global_step=epoch)
        writer.add_scalar('F1_score/val', F1_score, global_step=epoch)
        writer.add_scalar('MIoU/val', MIoU, global_step=epoch)
        writer.flush()

        # write train and validation results into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}\n" \
                         f"val_loss: {loss_val:.4f}\n" \
                         f"val_accuracy: {accuracy:.4f}\n" \
                         f"val_F1_score: {F1_score:.4f}\n" \
                         f"val_MIoU: {MIoU:.4f}\n"
            f.write(train_info + "\n\n")

        # reduce learning rate
        scheduler.step(loss_val)

    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
