import os
import errno
import numpy as np
from osgeo import gdal


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def loadImageData(path):
    '''
    load the remote sensing image data and transform it to numpy array
    Parameters:
        path        -- path of the image
    Returns:
        projection  -- projection information of the remote sensing image
        transform   -- coordinate system information of the remote sensing image
        all_data    -- numpy array representing the remote sensing image
    '''

    dataset = gdal.Open(path)
    assert dataset is not None, 'Image data does not exist!'

    # get row, column and band information of the remote sensing image
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()

    # create a numpy array to store the image data
    all_data = np.zeros((im_height, im_width, im_bands), dtype=float)
    for band_num in range(im_bands):
        band = dataset.GetRasterBand(band_num + 1)
        data = band.ReadAsArray(0, 0, im_width, im_height).astype(float)
        all_data[:, :, band_num] = data[:, :]

    return projection, transform, all_data


def arr2raster(arr, path, prj=None, trans=None):
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(path, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    dst_ds.GetRasterBand(1).WriteArray(arr)
    dst_ds.FlushCache()


def geo2imagexy(trans, x, y):
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x.astype(float) - trans[0], y.astype(float) - trans[3]])
    ans = np.linalg.solve(a, b)
    ans = ans.astype(int)

    return ans


def get_mean_std(img):
    # img shape: (h, w, channels)
    img = np.reshape(img, (-1, img.shape[-1]))
    mean = img.mean(axis=0)
    std = img.std(axis=0)
    return mean, std


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def Tiff64to8bit(img_64):
    if (np.max(img_64) - np.min(img_64) != 0):
        img_nrm = normalization(img_64)
        img_8 = np.uint8(255 * img_nrm)
        return img_8
    else:
        raise Exception('IMAGE has unique value, cannot be normalized!')
