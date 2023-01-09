import torch
import os
from PIL import Image
from torchvision.transforms import functional as F
from Res_UNet import ResUNet
from torch import nn
from torchvision import utils as vutils


def pre_trans(img, mean, std):
    img = F.to_tensor(img)
    img = F.normalize(img, mean=mean, std=std)
    return img


def seg(img, weights):
    model = ResUNet(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))["model"])

    # with open("model_ori.txt", "a") as f:
    #     for name, i in model.named_parameters():
    #         model_info = f"[{name:}\n{i}]"
    #         f.write(model_info + "\n")
    # model.eval()

    with torch.no_grad():
        outputs = model(img)
        softmax = nn.Softmax(dim=1)
        segmentation = torch.squeeze(softmax(outputs).argmax(dim=1).float())

    # with open("model_after.txt", "a") as f:
    #     for name, i in model.named_parameters():
    #         model_info = f"[{name:}\n{i}]"
    #         f.write(model_info + "\n")

    return segmentation


def save_seg(save_name, segmentation):
    vutils.save_image(segmentation, save_name)
    print(f'Segmentation saved as {save_name}')


def main():
    img_path = r"..\..\dataset\leftImg8bit\test\33.png"
    weights = r".\save_weights\best_model.pth"

    mean = [121.90957298, 130.36595824, 117.22788076]
    std = [38.19792449, 35.57061731, 41.42008869]

    img = Image.open(img_path)
    img = torch.unsqueeze(pre_trans(img, mean, std), 0)

    segmentation = seg(img, weights)
    save_name = os.path.basename(img_path).strip('.png')+'_seg.png'
    save_seg(save_name, segmentation)


if __name__ == '__main__':
    main()
