import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout,Concatenate, BatchNormalization, SeparableConv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, AveragePooling2D, Dense, concatenate, GlobalAveragePooling2D
import glob
from PIL import Image
from keras_unet_collection import models, losses
import model_utils as utils

def swim_transformer(inputs1=(512,512,3)):    
    transformer_model = models.swin_unet_2d(inputs1, filter_num_begin=64,
                               n_labels=1, depth=4, stack_num_down=2, stack_num_up=2,
                               patch_size=(4, 4), num_heads=[4, 8, 8, 8],
                               window_size=[4, 2, 2, 2], num_mlp=512, 
                               output_activation='Sigmoid', shift_window=True,
                               name='swin_unet')
    return transformer_model

def get_swim_tansformer(lr_rate, ip_size):
    model = swim_transformer((ip_size,ip_size,3))
    ADAM = Adam(learning_rate = lr_rate, clipnorm=1., clipvalue=0.5) 
    model.compile(optimizer = ADAM, loss = losses.dice	, metrics = [utils.dice_coef])
    return model
