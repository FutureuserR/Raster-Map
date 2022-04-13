from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os 
import cv2 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from sklearn.metrics import roc_curve, auc, precision_recall_curve 
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2019)

img_row = 256
img_col = 256
img_chan = 1
epochnum = 50
batchnum = 8
smooth = 1.
    
sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
adam = Adam(lr=1e-3) 

input_size = (img_row, img_col, img_chan)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
kinit = 'he_normal'

model = unet(adam, input_size, dice_loss)

#训练网络模型U-net
callbacks = [
#     EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
    ModelCheckpoint('model-T_unet-maproad.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
#u-net模型训练
results = model.fit(X_train, y_train, batch_size=1, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))
X_test, y_test = get_data(path_test, train=True)
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
preds_test = model.predict(X_test, verbose=1)
model.evaluate(X_test, y_test, batch_size=8, verbose=1)