# coding: utf-8
'''
    - train "ZF_UNET_224" CNN with random images
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__
from zf_unet_224_model import *

import glob
import data_factory, prepare_data_on_guangdong
import tifffile as tiff
import mkl
mkl.set_num_threads(6)
# def gen_random_image():
#     img = np.zeros((224, 224, 3), dtype=np.uint8)
#     mask = np.zeros((224, 224), dtype=np.uint8)
#
#     # Background
#     dark_color0 = random.randint(0, 100)
#     dark_color1 = random.randint(0, 100)
#     dark_color2 = random.randint(0, 100)
#     img[:, :, 0] = dark_color0
#     img[:, :, 1] = dark_color1
#     img[:, :, 2] = dark_color2
#
#     # Object
#     light_color0 = random.randint(dark_color0+1, 255)
#     light_color1 = random.randint(dark_color1+1, 255)
#     light_color2 = random.randint(dark_color2+1, 255)
#     center_0 = random.randint(0, 224)
#     center_1 = random.randint(0, 224)
#     r1 = random.randint(10, 56)
#     r2 = random.randint(10, 56)
#     cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
#     cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)
#
#     # White noise
#     density = random.uniform(0, 0.1)
#     for i in range(224):
#         for j in range(224):
#             if random.random() < density:
#                 img[i, j, 0] = random.randint(0, 255)
#                 img[i, j, 1] = random.randint(0, 255)
#                 img[i, j, 2] = random.randint(0, 255)
#
#     return img, mask
#
#
# def batch_generator(batch_size):
#     while True:
#         image_list = []
#         mask_list = []
#         for i in range(batch_size):
#             img, mask = gen_random_image()
#             image_list.append(img)
#             mask_list.append([mask])
#
#         image_list = np.array(image_list, dtype=np.float32)
#         if K.image_dim_ordering() == 'th':
#             image_list = image_list.transpose((0, 3, 1, 2))
#         image_list = preprocess_batch(image_list)
#         mask_list = np.array(mask_list, dtype=np.float32)
#         mask_list /= 255.0
#         yield image_list, mask_list

def load_train_file(path, batch):
    print(os.path.exists(os.path.join(path, "x_*.npy")))
    x_list = glob.glob(os.path.join(path, "mask_*.npy"))
    x_list = sorted(x_list)
    y_list = glob.glob(os.path.join(path, "mask_*.npy"))
    y_list = sorted(y_list)
    print(x_list)
    j=0
    while True:
        x = np.load(x_list[j % len(x_list)])
        y = np.load(y_list[j % len(y_list)])
        for i in range(500):
            xs, ys = data_factory.get_patches(x, y, batch)
            if K.image_dim_ordering() == "th":
                xs = np.transpose(xs, (0,3,1,2))
            yield xs, ys
            del xs, ys
        j+=1

def load_eval_file(path, batch):
    x = np.load(os.path.join(path, "x_val_for_building.npy"))
    y = np.load(os.path.join(path, "y_val_for_building.npy"))
    while True:
        xs, ys = data_factory.get_patches(x, y, batch)
        if K.image_dim_ordering() == "th":
            xs = np.transpose(xs, (0, 3, 1, 2))
        yield xs, ys


def train_iterator(train, label, batch):
    i = 0
    while True:
        image_list = []
        mask_list = []
        for j in range(batch):
            image_list.append(np.load(train[i % len(train)]))
            mask_list.append(np.load(label[i % len(label)]))
            i += 1
        image = np.array(image_list, dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image = np.transpose(image, (0, 3, 1, 2))
        mask = np.array(mask_list, dtype=np.float32)
        yield image, mask

def load_guangdong_data(path, batch):
    mask = tiff.imread(os.path.join(path, "bbox.tif")).astype(np.float32)
    mask_max = np.amax(mask)
    mask = mask / mask_max

    mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
    im_2017 = np.transpose(tiff.imread(os.path.join(path, "quickbird2017.tif")), (1,2,0))
    im_2017 = data_factory.stretch_n(im_2017)
    print(mask.shape, mask.max(),mask.min(),im_2017.shape, im_2017.max(), im_2017.min())
    while True:
        xs, ys = data_factory.get_patches(im_2017, mask, batch)
        if K.image_dim_ordering() == "th":
            xs = np.transpose(xs, (0, 3, 1, 2))
        yield xs, ys

def load_guangdong_test_data(path, batch):
    mask = tiff.imread(os.path.join(path, "tinysample.tif")).astype(np.float32)
    mask_max = np.amax(mask)
    mask = mask / mask_max
    print(mask.shape)

    im_2017 = np.transpose(tiff.imread(os.path.join(path, "quickbird2017.tif")), (1, 2, 0))
    im_2017 = data_factory.stretch_n(im_2017)

    xs, ys = data_factory.get_patches(im_2017, mask[:,:,:1], batch)
    if K.image_dim_ordering() == "th":
        xs = np.transpose(xs, (0, 3, 1, 2))
    return xs, ys

# def test_load_guangdong_data():
#     data_root = "/home/zj/PycharmProjects/tianchi/Dstl-Satellite-Imagery-Feature-Detection/dataset/"
#     # itr = load_guangdong_data(data_root, 24)
#     # xs, ys = next(itr)
#     xs, ys = load_guangdong_test_data(data_root,24)
#     import matplotlib.pyplot as plt
#     for i in range(24):
#         plt.ion()
#         for i in range(100):
#             plt.clf()
#             p1 = plt.subplot(121)
#             pl1 = p1.imshow(xs[i, :, :, :3])
#             # plt.colorbar()
#             p2 = plt.subplot(122)
#             pl2 = p2.imshow(ys[i, :, :, 0])
#             plt.draw()
#             plt.pause(1)
#         plt.show()
#         plt.ioff()



def train_unet(data_root):
    out_model_path = 'model/zf_unet_160_512.h5'
    epochs = 40
    patience = 10
    batch_size = 15
    optim_type = 'Adam'
    learning_rate = 0.001
    model = ZF_UNET_160()
    if os.path.isfile(out_model_path):
        model.load_weights(out_model_path)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('model/zf_unet_160_512_temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]



    # history = model.fit_generator(
    #     generator=load_train_file(data_root, batch_size),
    #     epochs=epochs,
    #     steps_per_epoch=50,
    #     validation_data=load_eval_file(data_root, batch_size),
    #     validation_steps=20,
    #     verbose=2,
    #     callbacks=callbacks)
    # x_test, y_test = load_guangdong_test_data(data_root, 240)
    # train_iter, valid_iter = prepare_data_on_guangdong.guangdong_dataset_iterator(
    #     data_root,batch_size)
    train_list = {"x":sorted(glob.glob(os.path.join(data_root, "train", "x_*.npy"))),
                  "y":sorted(glob.glob(os.path.join(data_root, "train", "mask_*.npy")))}
    valid_list = {"x":sorted(glob.glob(os.path.join(data_root, "test", "x_*.npy"))),
                  "y":sorted(glob.glob(os.path.join(data_root, "test", "mask_*.npy")))}
    print('Start training...')
    history = model.fit_generator(
        generator=train_iterator(train_list["x"],train_list["y"], batch_size),
        epochs=epochs,
        steps_per_epoch=50,
        validation_data=train_iterator(valid_list["x"], valid_list["y"], batch_size),
        validation_steps=7,
        verbose=1,
        callbacks=callbacks)
    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('zf_unet_224_train.csv')
    print('Training is finished (weights zf_unet_224.h5 and log zf_unet_224_train.csv are generated )...')

def generate_windows(data_path, window_size=160):
    im = np.transpose(tiff.imread(data_path), (1,2,0))
    im = data_factory.stretch_n(im)

    row = im.shape[0] // window_size
    col = im.shape[1] // window_size
    for i in range(row):
        for j in range(col):
            x = np.zeros((1,window_size,window_size,4)).astype(np.float32)
            x[0,:,:,:] = im[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size,:]
            if K.image_dim_ordering() == "th":
                x = np.transpose(x, (0,3,1,2))
            yield x

def generate_windows_2_change(data_root, window_size=160):
    im_17 = np.transpose(tiff.imread(os.path.join(data_root, "quickbird2017.tif")),(1,2,0))
    im_15 = np.transpose(tiff.imread(os.path.join(data_root, "quickbird2015.tif")),(1,2,0))
    im_17 = prepare_data_on_guangdong.stretch_n(im_17)
    im_15 = prepare_data_on_guangdong.stretch_n(im_15)
    im = im_17 - im_15
    del im_17
    del im_15
    row = im.shape[0] // window_size
    col = im.shape[1] // window_size
    for i in range(row):
        for j in range(col):
            x = np.zeros((1, window_size, window_size, 4)).astype(np.float32)
            x[0, :, :, :] = im[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size, :]
            if K.image_dim_ordering() == "th":
                x = np.transpose(x, (0, 3, 1, 2))
            yield x

def evaluation_on_guangdong_dataset(filename, name, c):
    windows =160
    model = ZF_UNET_160()
    weights = 'model/zf_unet_160_512_temp.h5'
    data_root = "dataset/"
    if os.path.exists(weights):
        model.load_weights(weights)
    mask = np.zeros((5106, 15106, 1)).astype(np.float32)
    if c:
        data_itr = generate_windows_2_change(data_root)
    else:
        data_itr = generate_windows(data_root + filename)
    i,j=0,0
    for im in data_itr:
        t_prd = model.predict_on_batch(im)
        mask[i*windows:(i+1)*windows, j*windows:(j+1)*windows, :] = t_prd[0,:,:,:]
        j += 1
        if j == (15106 // windows):
            j=0
            i+=1
    print(mask.max(), mask.min())
    tiff.imsave("predict/"+ name,mask)

def evaluation():
    model = ZF_UNET_160()
    weights = 'model/zf_unet_160_512_temp.h5'
    data_root = "dataset/data/change/"
    if os.path.exists(weights):
        model.load_weights(weights)

    # eval_itr = load_eval_file(data_root, 50)
    # eval_data, eval_mask = next(eval_itr)
    train_iter, valid_iter = prepare_data_on_guangdong.guangdong_dataset_iterator(
        data_root, 100)
    eval_data, eval_mask = next(valid_iter)
    t_prd = model.predict(eval_data, 16)
    if K.image_dim_ordering() == "th":
        eval_data = np.transpose(eval_data,(0,2,3,1))
    import matplotlib.pyplot  as plt
    plt.ion()
    for i in range(100):
        plt.clf()
        p1 = plt.subplot(131)
        pl1 = p1.imshow(eval_data[i,:,:,:3])
        # plt.colorbar()
        p2 = plt.subplot(132)
        pl2 = p2.imshow(eval_mask[i,:,:,0])

        p3 = plt.subplot(133)
        pl3 = p3.imshow(t_prd[i,:,:,0])
        plt.draw()
        plt.pause(1)
    plt.show()
    plt.ioff()

if __name__ == '__main__':
    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__
            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__
            print('Theano version: {}'.format(__theano_version__))
            import theano
            theano.config.openmp = True
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_dim_ordering())
    # data_root = "/home/zj/PycharmProjects/tianchi/Dstl-Satellite-Imagery-Feature-Detection/data/"
    data_root = "dataset/data/"
    # train_unet(data_root)
    # evaluation()
    evaluation_on_guangdong_dataset("quickbird2017.tif", "pred_change.tif", True)
    # evaluation_on_guangdong_dataset("quickbird2015.tif", "pred_15.tif")

    # test_load_guangdong_data()