import torch
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
from torchvision import utils as vutils


def eval(epoch, val_loader, model, criterion, device, num_classes):
    # unsolved: supposed to use eval while evaluating or testing, but got all zeros if add model.eval()
    #           I tried to record all the parameters of the model before and after predicting, and the parameters
    #           corresponding with model.eval(), however the four files are exactly the same.

    # model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            target = nn.functional.one_hot(target, num_classes)
            target = target.permute(0, 3, 1, 2)

            # loss
            outputs = model(inputs)
            target = target.argmax(dim=1).long()
            loss = criterion(outputs, target)
            print("val: ", epoch, i, loss.item())

            # test
            # softmax = nn.Softmax(dim=1)
            # segmentation = torch.squeeze(softmax(outputs).argmax(dim=1).float())
            # save_name = './p/seg_' + str(epoch) + str(i) + '.png'
            # vutils.save_image(segmentation, save_name)

            # Confusion Matrix
            softmax = nn.Softmax(dim=1)
            segmentation = softmax(outputs).argmax(dim=1)
            cf_matrix = confusion_matrix(torch.flatten(target).cpu(), torch.flatten(segmentation).cpu())
            plt.figure(figsize=(12, 7))
            sn.heatmap(cf_matrix, annot=True)
            plt.savefig(f'Confusion_Matrix.png')
            # plt.savefig(f'./c/Confusion_Matrix_{epoch}{i}.png')
            plt.close()

            # Accuracy
            accuracy = np.diag(cf_matrix).sum() / cf_matrix.sum()

            # F1_score
            F1_score = 2 * cf_matrix[1, 1] / (2 * cf_matrix[1, 1] + cf_matrix[0, 1] + cf_matrix[1, 0])

            # MIoU
            MIoU = np.diag(cf_matrix) / (cf_matrix.sum() - np.flipud(np.diag(cf_matrix)))
            MIoU = np.nanmean(MIoU)

    return loss, accuracy, F1_score, MIoU
