import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras.layers as L
import model_utils as utils

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = L.MaxPool2D((2, 2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv2D(num_filters, 1, padding="same")(out)
    out = L.Activation("sigmoid")(out)

    return out * s



def PSA_block(saliency_map, f, num_filters):
    saliency_map = L.AveragePooling2D(pool_size=(2, 2))(saliency_map)
    s_ip = L.Conv2D(num_filters, 1, padding="same")(saliency_map)
    s_ip = L.BatchNormalization()(s_ip)

    f_ip = L.Conv2D(num_filters, 1, padding="same")(f)
    f_ip = L.BatchNormalization()(f_ip)

    An = L.Activation("relu")(s_ip + f_ip)
    An = L.Conv2D(num_filters, 1, padding="same")(An)
    sp_co = L.Activation("sigmoid")(An)
    out = f*sp_co

    return out,saliency_map
   

def decoder_block(x, s, num_filters):
    x = L.UpSampling2D(interpolation="bilinear")(x)
    s = attention_gate(x, s, num_filters)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def attention_unet(inputs1=(512,512,3)):
    """ Inputs """
    inputs1 = L.Input((512,512,3))
    """ Encoder """

    s1,p1 = encoder_block(inputs1,64)

    s2, p2 = encoder_block(p1, 128)

    s3, p3 = encoder_block(p2, 256)

    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    """ Outputs """
    outputs = L.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    """ Model """
    model = Model(inputs1, outputs, name="Attention-UNET")
    return model

def cascaded_attention_unet(inputs1=(512,512,3),inputs2=(512,512,1)):
    """ Inputs """
    inputs1 = L.Input((512,512,3))
    inputs2 = L.Input((512,512,1))

    """ Encoder """
    s1,p1 = encoder_block(inputs1,64)
    a1,c1 = PSA_block(inputs2,p1,64)
    

    s2, p2 = encoder_block(a1, 128)
    a2,c2 = PSA_block(c1,p2,128)

    s3, p3 = encoder_block(a2, 256)
    a3,c3 = attention_gate(c2,p3,256)

    s4, p4 = encoder_block(a3, 512)
    a4,c4 = attention_gate(c3,p4,512)

    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    """ Outputs """
    outputs = L.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    """ Model """
    model = Model([inputs1,inputs2], outputs, name="Cascaded-Attention-UNET")
    return model


def get_AUnet(lr_rate, ip_size):
    model = attention_unet((ip_size,ip_size,3))
    ADAM = Adam(learning_rate = lr_rate, clipnorm=1., clipvalue=0.5) 
    model.compile(optimizer = ADAM, loss = utils.dice_coef_loss	, metrics = [utils.dice_coef])
    return model

def Cas_get_AUnet(lr_rate, ip_size):
    model = cascaded_attention_unet((ip_size,ip_size,3),(ip_size,ip_size,3))
    ADAM = Adam(learning_rate = lr_rate, clipnorm=1., clipvalue=0.5) 
    model.compile(optimizer = ADAM, loss = utils.dice_coef_loss	, metrics = [utils.dice_coef])
    return model