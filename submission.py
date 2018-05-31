import numpy as np
import tifffile as tiff
import prepare_data_on_guangdong
import keras.backend as K
import os

def generate_windows(data_path, window_size=160):
    im = np.transpose(tiff.imread(data_path), (1,2,0))
    im = prepare_data_on_guangdong.stretch_n(im)

    row = im.shape[0] // window_size
    col = im.shape[1] // window_size
    for i in range(row):
        for j in range(col):
            x = np.zeros((1,window_size,window_size,4)).astype(np.float32)
            x[0,:,:,:] = im[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size,:]
            if K.image_dim_ordering() == "th":
                x = np.transpose(x, (0,3,1,2))
            yield x

def get_predict(model, datapath, files):
    windows =160
    model = ZF_UNET_160()
    weights = "zf_unet_224_temp.h5"
    data_root = "/home/zj/PycharmProjects/tianchi/Dstl-Satellite-Imagery-Feature-Detection/dataset/"
    if os.path.exists(weights):
        model.load_weights(weights)
    mask = np.zeros((5106, 15106, 1)).astype(np.float32)
    data_itr = generate_windows(datapath)
    i,j=0,0
    for im in data_itr:
        t_prd = model.predict_on_batch(im)
        mask[i*windows:(i+1)*windows, j*windows:(j+1)*windows, :] = t_prd[0,:,:,:]
        j += 1
        if j == (15106 // windows):
            j=0
            i+=1
    tiff.imsave(files, mask)
    return mask


def evaluation_on_guangdong_dataset():
    pass

def change_dection(mask1, mask2, threhold1, threhold2):
    mask1 = np.where(mask1>threhold1, 1, 0).astype(np.float32)
    mask2 = np.where(mask2 > threhold2, 1, 0).astype(np.float32)
    change = mask1 - mask2
    result = np.where(change>0, 1, 0).astype(np.float32)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    tiff.imsave("results/result_%g_%g.tif" % (threhold1,threhold2), result)

if __name__ == "__main__":
    # mask1 = tiff.imread("pred_2017.tif")
    # mask2 = tiff.imread("predict/pred_15.tif")
    # tiff.imsave("results/change_sub.tif", mask1-mask2);
    # tr1= [0.8,0.9,0.92,0.94]
    # tr2=[0.4,0.5,0.6]
    # for thred2 in tr2:
    #     for thred1 in tr1:
    #         change_dection(mask1, mask2, thred1, thred2)
    mask2 = tiff.imread("predict/pred_change.tif")
    mask2 = np.where(mask2 >0.97, 1, 0).astype(np.float32)
    tiff.imsave("results/result_0_9.tif", mask2)


