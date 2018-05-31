# ZF_UNET_224 Pretrained Model
Modification of convolutional neural net "UNET" for image segmentation in Keras framework

## Requirements

Python 3.*, Keras 2.0.8, Theano 0.9 or Tensorflow 1.3.0

## Usage

```python
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef
from keras.optimizers import Adam

model = ZF_UNET_224()
model.load_weights("zf_unet_224.h5") # optional
optim = Adam()
model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

model.fit(...)
```

