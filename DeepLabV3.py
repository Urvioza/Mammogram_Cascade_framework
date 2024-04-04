
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
from keras.applications.resnet50 import ResNet50
import model_utils as utils

class ConvBlock(Layer):
    
    def __init__(self, filters=256//4, kernel_size=3, use_bias=False, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        
        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', dilation_rate=dilation_rate, use_bias=use_bias, kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
    
    def call(self, X): return self.net(X)        
        
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "kernel_size":self.kernel_size,
            "use_bias":self.use_bias,
            "dilation_rate":self.dilation_rate
        }
    
def AtrousSpatialPyramidPooling(X):
    
    # Shapes 
    _, height, width, _ = X.shape
    
    # Image Pooling 
    image_pool = AveragePooling2D(pool_size=(height, width), name="ASPP-AvgPool2D")(X)
    image_pool = ConvBlock(kernel_size=1, name="ASPP-ConvBlock-1")(image_pool)
    image_pool = UpSampling2D(size=(height//image_pool.shape[1], width//image_pool.shape[2]), name="ASPP-UpSampling")(image_pool)
    
    # Conv Blocks
    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-Conv-1")(X)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-Conv-6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-Conv-12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-Conv-18")(X)
    
    # Concat All
    concat = Concatenate(axis=-1, name="ASPP-Concat")([image_pool, conv_1, conv_6, conv_12, conv_18])
    net = ConvBlock(kernel_size=1, name="ASPP-Net")(concat)
    
    return net



def deeplabv3_plus(IMAGE_SIZE):
    # Input
    InputL = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="InputLayer")
   
    # Base Mode
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=InputL)
    
    # ASPP Phase
    DCNN = resnet50.get_layer('conv4_block6_2_relu').output
    ASPP = AtrousSpatialPyramidPooling(DCNN)
    ASPP = UpSampling2D(size=(IMAGE_SIZE//4//ASPP.shape[1], IMAGE_SIZE//4//ASPP.shape[2]), name="AtrousSpatial")(ASPP)
    
    # LLF Phase
    LLF = resnet50.get_layer('conv2_block3_2_relu').output
    LLF = ConvBlock(filters=48//4, kernel_size=1, name="LLF-ConvBlock")(LLF)
    
    # Combined
    combined = Concatenate(axis=-1, name="Combine-LLF-ASPP")([ASPP, LLF])
    features = ConvBlock(name="Top-ConvBlock-1")(combined)
    features = ConvBlock(name="Top-ConvBlock-2")(features)
    upsample = UpSampling2D(size=(IMAGE_SIZE//features.shape[1], IMAGE_SIZE//features.shape[1]), interpolation='bilinear', name="Top-UpSample")(features)
    
    # Output Mask
    PredMask = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid', use_bias=False, name="OutputMask")(upsample)
    
    # DeelLabV3+ Model
    model = Model(InputL, PredMask, name="DeepLabV3-Plus")
    return model

def get_DLV3(lr_rate, ip_size):
    model = deeplabv3_plus(ip_size)
    ADAM = Adam(learning_rate = lr_rate, clipnorm=1., clipvalue=0.5) 
    model.compile(optimizer = ADAM, loss = utils.dice_coef_loss	, metrics = [utils.dice_coef])
    return model