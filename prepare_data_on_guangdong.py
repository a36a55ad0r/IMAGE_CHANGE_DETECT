import numpy as np
import tifffile as tiff
import numpy.random as random
import cv2
import matplotlib.pyplot as plt
import os, glob
WINDOW = 160
STRIDE = 64


def load_data():
    im_2017 = np.transpose(tiff.imread("dataset/quickbird2017.tif"), (1, 2, 0))
    im_2015 = tiff.imread("dataset/quickbird2015.tif").transpose(1, 2, 0)
    tinymap = tiff.imread("dataset/answer_complete.tif").astype(np.float32)
    # tinymax = np.amax(tinymap)
    # tinymap = tinymap / tinymax

    print(im_2017.shape, tinymap.shape)
    return im_2017, im_2015, tinymap

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


def generate_windows_from_mask(aug=True):
    im_17, im_15, tinymap = load_data()
    im_17 = stretch_n(im_17)
    im_15 = stretch_n(im_15)

    row = (im_17.shape[0] - WINDOW) // STRIDE + 1
    col = (im_17.shape[1] - WINDOW) // STRIDE + 1

    tr = 0.2
    if not os.path.exists("dataset/data/change/"):
        os.makedirs("dataset/data/change/")
    for i in xrange(row):
        for j in xrange(col):
            mask = np.zeros((WINDOW, WINDOW, 1)).astype(np.float32)
            mask[:,:,0] = tinymap[i * STRIDE: i * STRIDE + WINDOW, j * STRIDE: j * STRIDE + WINDOW, 0]

            rate = np.sum(mask) / np.size(mask)
            if rate >= tr:
                # x = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 4)).astype(np.float32)
                x1 = im_17[i * STRIDE: i * STRIDE + WINDOW, j * STRIDE: j * STRIDE + WINDOW, :]
                x2 = im_15[i * STRIDE: i * STRIDE + WINDOW, j * STRIDE: j * STRIDE + WINDOW, :]
                x = np.asarray(x1 -x2).astype(np.float32)
                np.save("dataset/data/change/x_building_change_window_%.4d_%.4d.npy" % (i, j), x)
                np.save("dataset/data/change/mask_building_change_window_%.4d_%.4d.npy" % (i, j), mask)
                if aug is True:
                    if random.uniform(0, 1) > 0.5:
                        x = x[::-1]
                        mask = mask[::-1]
                        np.save("dataset/data/change/x_building_change_window_%.4d_%.4d_top_down.npy" % (i, j), x)
                        np.save("dataset/data/change/mask_building_change_window_%.4d_%.4d_top_down.npy" % (i, j), mask)
                    if random.uniform(0, 1) > 0.5:
                        x = x[:, ::-1]
                        mask = mask[:, ::-1]
                        np.save("dataset/data/change/x_building_change_window_%.4d_%.4d_left_right.npy" % (i, j), x)
                        np.save("dataset/data/change/mask_building_change_window_%.4d_%.4d_left_right.npy" % (i, j), mask)

            else: continue


def generate_windows(image, masks, name, aug=True):
    row = (image.shape[0] - WINDOW) // STRIDE + 1
    col = (image.shape[1] - WINDOW) // STRIDE + 1

    tr = 0.2
    if not os.path.exists("dataset/data"):
        os.makedirs("dataset/data")
    for i in xrange(row):
        for j in xrange(col):
            mask = np.zeros((WINDOW, WINDOW, 1)).astype(np.float32)
            mask[:, :, 0] = masks[i * STRIDE: i * STRIDE + WINDOW, j * STRIDE: j * STRIDE + WINDOW]
            rate = np.sum(mask) / np.size(mask)
            if rate >= tr:
                x = np.zeros((WINDOW, WINDOW, 4)).astype(np.float32)
                x[:,:,:] = image[i * STRIDE: i * STRIDE + WINDOW, j * STRIDE: j * STRIDE + WINDOW, :]
                if i<row//5:
                    np.save("dataset/data/test/x_%s_%.4d_%.4d.npy" % (name, i, j), x)
                    np.save("dataset/data/test/mask_%s_%.4d_%.4d.npy" % (name, i, j), mask)
                else:
                    np.save("dataset/data/train/x_%s_%.4d_%.4d.npy" % (name, i, j), x)
                    np.save("dataset/data/train/mask_%s_%.4d_%.4d.npy" % (name, i, j), mask)

                if aug is True:
                    if random.uniform(0, 1) > 0.5:
                        x = x[::-1]
                        mask = mask[::-1]
                        if i < row // 5:
                            np.save("dataset/data/test/x_%s_%.4d_%.4d_top_down.npy" % (name, i, j), x)
                            np.save("dataset/data/test/mask_%s_%.4d_%.4d_top_down.npy" % (name, i, j), mask)
                        else:
                            np.save("dataset/data/train/x_%s_%.4d_%.4d_top_down.npy" % (name, i, j), x)
                            np.save("dataset/data/train/mask_%s_%.4d_%.4d_top_down.npy" % (name, i, j), mask)
                    if random.uniform(0, 1) > 0.5:
                        x = x[:, ::-1]
                        mask = mask[:, ::-1]
                        if i < row // 5:
                            np.save("dataset/data/test/x_%s_%.4d_%.4d_left_right.npy" % (name, i, j), x)
                            np.save("dataset/data/test/mask_%s_%.4d_%.4d_left_right.npy" % (name, i, j), mask)
                        else:
                            np.save("dataset/data/train/x_%s_%.4d_%.4d_left_right.npy" % (name, i, j), x)
                            np.save("dataset/data/train/mask_%s_%.4d_%.4d_left_right.npy" % (name, i, j), mask)

            else:
                continue

def prepare_data():
    data_root = "dataset/"
    im_17, im_15, masks = load_data()
    im_17 = stretch_n(im_17)
    im_15 = stretch_n(im_15)
    im = im_17 -im_15
    generate_windows(im, np.where(masks==1, 1, 0),"positive")
    generate_windows(im, np.where(masks==2, 1, 0),"negetive")


def visualize_data():
    x_list = glob.glob("dataset/data/x_*.npy")
    mask_list = glob.glob("dataset/data/mask_*.npy")
    x_list = sorted(x_list)
    mask_list = sorted(mask_list)
    print(x_list)
    plt.ion()
    for i in range(len(x_list)):
        x = np.load(x_list[i])
        mask = np.load(mask_list[i])
        plt.cla()
        p1 = plt.subplot(121)
        xpl = p1.imshow(x[:,:,:3])

        p2 = plt.subplot(122)
        maskplt = p2.imshow(mask[:,:,0])
        plt.draw()
        plt.pause(1)
    plt.ioff()


# def merge_mask():
#     data_root = "dataset/"
#     mask_17 = tiff.imread(data_root+"mask_2015_new.tif").astype(np.int32)
#     mask_17_old = tiff.imread(data_root+"bbox_15.tif").astype(np.int32)
#
#     mask = (mask_17 | mask_17_old)
#     mask = mask.astype(np.float32)
#     print(mask.shape)
#     tiff.imsave("dataset/mask_2015.tif", mask)

def split_dataset(path):
    x_list = glob.glob(os.path.join(path, "x_*.npy"))
    y_list = glob.glob(os.path.join(path, "mask_*.npy"))
    x_list = sorted(x_list)
    y_list = sorted(y_list)
    # print(x_list)
    # print(y_list)
    shuffle_index = np.random.permutation(len(x_list))
    test_num = len(x_list) // 5
    test_x, test_y= [], []
    train_x, train_y= [],[]
    for i in range(test_num):
        test_x.append(x_list[shuffle_index[i]])
        test_y.append(y_list[shuffle_index[i]])

    for i in range(test_num,len(x_list)):
        train_x.append(x_list[shuffle_index[i]])
        train_y.append(y_list[shuffle_index[i]])
    print(train_y)
    print(test_y)
    return train_x, train_y, test_x, test_y

def guangdong_dataset_iterator(path, batch):
    import keras.backend as K
    train_x, train_y, test_x, test_y = split_dataset(path)
    print(len(train_x))
    print(len(test_y))
    def train_iterator(train, label, batch):
        i=0
        while True:
            image_list =[]
            mask_list = []
            for j in range(batch):
                image_list.append(np.load(train[i % len(train)]))
                mask_list.append(np.load(label[i % len(label)]))
                i+=1
            image = np.array(image_list, dtype=np.float32)
            if K.image_dim_ordering()=='th':
                image = np.transpose(image,(0,3,1,2))
            mask = np.array(mask_list,dtype=np.float32)
            yield image, mask


    train_iter = train_iterator(train_x,train_y,batch)
    valid_iter = train_iterator(test_x, test_y,batch)
    return train_iter, valid_iter


if __name__ == "__main__":
    # merge_mask()

    # generate_windows_from_mask()
    prepare_data()
    # train_iter, valid_iter = guangdong_dataset_iterator("dataset/data/change",12)
    # for i in range(2000):
    #     x, y = next(train_iter)
    #     print(x.shape, y.shape)
    # visualize_data()