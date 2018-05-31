# -*-coding:utf-8 -*-
import numpy as np
import numpy.random as random
import cv2
from shapely.wkt import loads as wkt_loads
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import time
data_root = "/home/zj/PycharmProjects/tianchi/Dstl-Satellite-Imagery-Feature-Detection"
TW= pd.read_csv(data_root+"/data/train_wkt_v4.csv")
GS = pd.read_csv(data_root+"/data/grid_sizes.csv", names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
PIXELS_SIZE =1600
ISZ =160
TRAIN_IMAGE_SIZE = 20
EVAL_IMAGE_SIZE = 5
DIM = 4

def _convert_coordinates_to_raster(coords, img_size, xymax):
    # 由图片的地理坐标转换为像素坐标， 对齐
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # 获得图片的地理坐标最大最小值
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # 从 train_wkt文件得到图像cType标记类别的多边形标记区域
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    #
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=TW):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask

def load_image(id, dim):
    # 载入rgb或者rgb nir图片
    if dim==3:
        img = tiff.imread(data_root+"/data/three_band/{}.tif" % id)
        img = cv2.resize(img, (PIXELS_SIZE, PIXELS_SIZE))
    elif dim == 4:
        img_RGB = np.transpose(tiff.imread(data_root+"/data/three_band/{}.tif".format(id)),(1,2,0))
        img_RGB = cv2.resize(img_RGB, (PIXELS_SIZE, PIXELS_SIZE))
        img_nir1 = np.transpose(tiff.imread(data_root+"/data/sixteen_band/{}_M.tif".format(id)),(1,2,0))[:,:, 6]
        img_nir1 = cv2.resize(img_nir1, (PIXELS_SIZE, PIXELS_SIZE))
        img = np.zeros((img_RGB.shape[0], img_RGB.shape[1], 4)).astype(np.float32)
        img[:, :, 0:3] = img_RGB
        img[:, :, 3] = img_nir1
    return img

def stretch_n(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


def generate_image_and_mask(imageId, classtype):
    img = load_image(imageId, DIM)
    img = stretch_n(img)
    print(img.shape, imageId, np.amax(img), np.amin(img))
    mask = generate_mask_for_image_and_class((img.shape[0], img.shape[1]), imageId, class_type=classtype)
    return img, mask


def prepare_building_data():
    print("start to prepare building data")
    print("train image size:{} validation image size:")
    ids = TW.ImageId.unique()  # get all image file name
    x = np.zeros((2*PIXELS_SIZE, 2*PIXELS_SIZE, DIM)).astype(np.float32)
    y = np.zeros((2*PIXELS_SIZE, 2*PIXELS_SIZE, 1)).astype(np.float32)
    for k in range(5):
        for i in range(2):
            for j in range(2):
                id = ids[4*k + 2*i +j]

                img, mask = generate_image_and_mask(id, classtype=1)  # get file{id} normalized img and building mask
                x[i*PIXELS_SIZE:(i+1)*PIXELS_SIZE, j*PIXELS_SIZE:(j+1)*PIXELS_SIZE,:] = img[:PIXELS_SIZE, :PIXELS_SIZE,:]
                y[i*PIXELS_SIZE:(i+1)*PIXELS_SIZE, j*PIXELS_SIZE:(j+1)*PIXELS_SIZE,0] = mask[:PIXELS_SIZE, :PIXELS_SIZE]
        np.save(data_root+"/data/x_trn_for_building_{}.npy".format(k), x)
        np.save(data_root+"/data/y_trn_for_building_{}.npy".format(k), y)

    x = np.zeros((5*PIXELS_SIZE, PIXELS_SIZE,DIM)).astype(np.float32)
    y = np.zeros((5*PIXELS_SIZE, PIXELS_SIZE,1)).astype(np.float32)
    for k in range(5):
        id = ids[20+k]
        img, mask = generate_image_and_mask(id, classtype=1)  # get file{id} normalized img and building mask
        x[k*PIXELS_SIZE:(k+1)*PIXELS_SIZE,:PIXELS_SIZE,:] = img[:PIXELS_SIZE,:PIXELS_SIZE,:]
        y[k*PIXELS_SIZE:(k+1)*PIXELS_SIZE,:PIXELS_SIZE,0] = mask[:PIXELS_SIZE,:PIXELS_SIZE]
        np.save(data_root+"/data/x_val_for_building.npy", x)
        np.save(data_root+"/data/y_val_for_building.npy", y)

def get_patches(img, msk, amt=10000, aug=True):

    start = time.clock()
    is2 = int(1.0 * ISZ)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    # tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    tr = [0.2]
    while True:
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(1):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)
        if(len(x) >= amt):
            break;
    # x, y = np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 1, 2, 3))
    x, y= 2.0 * np.asarray(x) - 1.0, np.asarray(y)
    # print("data generated cost {:.2f}".format((time.clock() - start) / 60))
    # print x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y)
    return x, y

if __name__ == "__main__":
    prepare_building_data()
    img = np.load("../data/x_val_for_building.npy")
    mask = np.load("../data/y_val_for_building.npy")

    import time

    start = time.clock()
    x, y = get_patches(img, mask, 1000)
    plt.ion()
    for i in range(20):
        plt.clf()

        p1 = plt.subplot(121)
        img_plt = p1.imshow(np.transpose(x[10*i],(0,1,2))[:,:,:3])


        p2 = plt.subplot(122)
        m = np.transpose(y[10*i], (0,1, 2))
        mask_plt = p2.imshow(m[:,:,0])
        plt.draw(); plt.pause(1)

    plt.show()
    plt.ioff()
    print("finished in {:.2f}".format((time.clock() -start)/60))