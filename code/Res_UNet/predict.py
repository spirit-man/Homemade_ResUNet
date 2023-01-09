import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
from utils import loadImageData, arr2raster, geo2imagexy, get_mean_std
from test import pre_trans, seg
from sklearn import decomposition
from sklearn.metrics import confusion_matrix


def loadSampleData(path):
    reader = pd.ExcelFile(path)
    sheet_names = reader.sheet_names
    # get columns
    header = pd.read_excel(path, sheet_name=sheet_names[0]).columns
    # create an empty dataframe to store all the next dataframes in the iteration
    alldf = pd.DataFrame(columns=header)
    for sheet in sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        alldf = pd.concat([alldf, df])

    return alldf


def get_RGB(img, type: bool):
    if type:
        # Landsat 2007(band 3, 2, 1)
        return img[:, :, 3:0:-1]
    else:
        # Landsat 2013 and 2022(band 4, 3, 2)
        return img[:, :, 4:1:-1]


def predict(path: str, weights):
    # Remember to clip your Landsat image and make sure it does not have NoData value,
    # otherwise normalization will return all black or white image!!!

    prj, trans, landsat = loadImageData(path)
    landsat.astype(int)

    # use true RGB to predict
    type = '2007' in path
    true_rgb = get_RGB(landsat, type)
    mean_rgb, std_rgb = get_mean_std(true_rgb)
    true_rgb = torch.unsqueeze(pre_trans(true_rgb.copy(), mean_rgb, std_rgb), 0)
    segmentation_rgb = seg(true_rgb.to(torch.float32), weights)

    # use PCA to get 3 channels and then predict
    pca = decomposition.PCA(n_components=3)
    pca_features = pca.fit_transform(landsat.reshape((-1, landsat.shape[-1])))
    pca_features = pca_features.reshape((landsat.shape[0], landsat.shape[1], -1))
    mean_pca, std_pca = get_mean_std(pca_features)
    pca_features = torch.unsqueeze(pre_trans(pca_features.copy(), mean_pca, std_pca), 0)
    segmentation_pca = seg(pca_features.to(torch.float32), weights)

    # save as tiff
    result_rgb = segmentation_rgb.numpy()
    save_name_rgb = os.path.basename(path).strip('.tif') + '_seg_rgb.tif'
    result_path_rgb = os.path.join(os.path.dirname(path), save_name_rgb)
    arr2raster(arr = result_rgb, path = result_path_rgb, prj = prj, trans = trans)
    print(f'RGB Segmentation saved as {save_name_rgb}')

    result_pca = segmentation_pca.numpy()
    save_name_pca = os.path.basename(path).strip('.tif') + '_seg_pca.tif'
    result_path_pca = os.path.join(os.path.dirname(path), save_name_pca)
    arr2raster(arr = result_pca, path = result_path_pca, prj = prj, trans = trans)
    print(f'PCA Segmentation saved as {save_name_pca}')

    return result_path_rgb, result_path_pca


def evaluate(result_path, sample_path):
    prj, trans, result = loadImageData(result_path)
    result = np.squeeze(result).astype(int)
    sample = loadSampleData(sample_path)

    # calculate the image coordinate of sample points
    x, y = map(np.array, [sample.iloc[:, 1], sample.iloc[:, 2]])
    sample_img_coord = geo2imagexy(trans, x, y)

    # find the corresponding value of result image
    result_sample = result[sample_img_coord.tolist()]
    result_sample = result_sample.flatten()

    # calculate confusion matrix and total accuracy
    cf_matrix = confusion_matrix(result_sample.tolist(), np.ones(result_sample.shape, dtype=int).tolist())
    plt.figure(figsize=(12, 7))
    sn.heatmap(cf_matrix, annot=True)
    cf_matrix_path = os.path.join(os.path.dirname(result_path), os.path.basename(result_path).strip('.tif') + '_Confusion_Matrix.png')
    plt.savefig(cf_matrix_path)
    plt.close()

    accuracy = np.diag(cf_matrix).sum() / cf_matrix.sum()
    print(f'Accuracy on sample points: {accuracy}')


def main():
    img_path = r"..\..\Landsat\Landsat2022.tif"
    weights = r".\save_weights\best_model.pth"
    sample_path = r"..\..\Landsat\2022_Bluegreen_algae.xlsx"

    result_path_rgb, result_path_pca = predict(img_path, weights)
    evaluate(result_path_rgb, sample_path)
    evaluate(result_path_pca, sample_path)


if __name__ == '__main__':
    main()
