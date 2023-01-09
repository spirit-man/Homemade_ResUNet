import os
from torch.utils.data import Dataset
from PIL import Image


class BlueAlgaeDataset(Dataset):
    def __init__(self, filepath: str, train: int, transforms=None):
        super(BlueAlgaeDataset, self).__init__()
        assert os.path.exists(filepath), f"path '{filepath}' does not exists."
        self.transforms = transforms
        # parameter train stands for: 0(train) or 1(val) or 2(test)
        assert train == 0 or train == 1 or train == 2, "please pass 0(train) or 1(val) or 2(test)!"
        if train == 0:
            img_dir = os.path.join(filepath, "leftImg8bit", "train")
            mask_dir = os.path.join(filepath, "gtFine", "train")
        elif train == 1:
            img_dir = os.path.join(filepath, "leftImg8bit", "val")
            mask_dir = os.path.join(filepath, "gtFine", "val")
        else:
            img_dir = os.path.join(filepath, "leftImg8bit", "test")
            mask_dir = os.path.join(filepath, "gtFine", "test")

        img_names = [i for i in os.listdir(img_dir) if i.endswith(".png")]
        self.img_list = [os.path.join(img_dir, i) for i in img_names]
        self.mask = [os.path.join(mask_dir, i.strip(".png") + "_gtFine_labelIds.png") for i in img_names]

        for m in self.mask:
            if os.path.exists(m) is False:
                raise FileNotFoundError(f"file {m} does not exists.")

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        mask = Image.open(self.mask[index])
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        else:
            raise Exception("Transform cannot be None! Do ToTensor and Normalize at least!")
        return img, mask

    def __len__(self):
        return len(self.img_list)
