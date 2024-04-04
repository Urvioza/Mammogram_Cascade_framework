import numpy as np
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import scipy.misc as sc
import cv2
import random
import glob
import shutil
import pandas as pd
from skimage import exposure

def get_train_test_fold(set1,set2,set3,set4,set5):
    Train_Images = []
    Train_Masks = []
    Test_Images = []
    Test_Masks = []

    Test_Images.append(pd.DataFrame(set1,columns =['Path']))
    Train_Images.append(pd.DataFrame(np.concatenate((set2,set3,set4,set5)),columns =['Path']))


    Test_Images.append(pd.DataFrame(set2,columns =['Path']))
    Train_Images.append(pd.DataFrame(np.concatenate((set1,set3,set4,set5)),columns =['Path']))


    Test_Images.append(pd.DataFrame(set3,columns =['Path']))
    Train_Images.append(pd.DataFrame(np.concatenate((set2,set1,set4,set5)),columns =['Path']))


    Test_Images.append(pd.DataFrame(set4,columns =['Path']))
    Train_Images.append(pd.DataFrame(np.concatenate((set2,set3,set1,set5)),columns =['Path']))


    Test_Images.append(pd.DataFrame(set5,columns =['Path']))
    Train_Images.append(pd.DataFrame(np.concatenate((set2,set3,set4,set1)),columns =['Path']))

    return Train_Images, Test_Images


def truncation_normalization(img, mask):
    Pmin = np.percentile(img[mask!=0], 5)
    Pmax = np.percentile(img[mask!=0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[mask==0]=0
    return normalized

def clahe(img, clip):
    cl = exposure.equalize_adapthist(img, clip_limit= clip)
    return cl

def normalize_image_clahe(image):
    image = image.astype(np.uint8)
    blur = cv2.GaussianBlur(image,(5,5),0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    normalized = truncation_normalization(image, breast_mask)
    cl1 = clahe(normalized, 0.02)
    cl2 = clahe(normalized, 0.03)
    synthetized = cv2.merge((np.array(normalized),cl1,cl2))
    return synthetized

def preprocess_samples(img):
  a = np.expand_dims(img[:,:,0], axis=-1)
  norm = normalize_image_clahe(a)
  return norm


def preprocess_masks(mask):
  norm_mask = mask /255
  norm_mask[norm_mask > 0.1] = 1
  norm_mask[norm_mask <= 0.1] = 0
  return norm_mask

def preprocess_seg(mask):
  norm_mask = mask /255
  return norm_mask


datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.3,
    fill_mode='nearest',
    horizontal_flip=True,
    preprocessing_function= preprocess_samples)

mask_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.3,
    fill_mode='nearest',
    horizontal_flip=True,
    preprocessing_function= preprocess_masks)

segmenter_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.3,
    fill_mode='nearest',
    horizontal_flip=True,
    preprocessing_function= preprocess_seg
    )

val_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_samples
    )

val_mask_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_masks
    )

val_segmenter_datagen = ImageDataGenerator(
   preprocessing_function= preprocess_seg
    )


def generate_dataframe(dframe_I,dframe_M, img_path, mask_path, t_batch = 2):
  #print(dframe_M)
  image_generator=datagen.flow_from_dataframe(
  dframe_I,
  directory = img_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "rgb",
  batch_size=t_batch,
  seed= 1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  Mask_generator=mask_datagen.flow_from_dataframe(
  dframe_M,
  directory =mask_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "grayscale",
  batch_size=t_batch,
  seed=1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  train_generator = zip(image_generator,Mask_generator)
  return train_generator


def generate_valid_dataframe_cascade(dframe_val_I,dframe_val_M, img_path, mask_path, v_batch = 8):

  Val_image_generator=val_datagen.flow_from_dataframe(
  dframe_val_I,
  directory = img_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "rgb",
  batch_size=v_batch,
  seed= 1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )


  Val_Mask_generator=val_mask_datagen.flow_from_dataframe(
  dframe_val_M,
  directory = mask_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "grayscale",
  batch_size=v_batch,
  seed=1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  val_generator = zip(Val_image_generator, Val_Mask_generator)
  return val_generator


def generate_dataframe_cascade(dframe_I,dframe_M, img_path, mask_path, PSA_ip_path, t_batch = 2):
  #print(dframe_M)
  image_generator=datagen.flow_from_dataframe(
  dframe_I,
  directory = img_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "rgb",
  batch_size=t_batch,
  seed= 1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  pre_segment_generator=segmenter_datagen.flow_from_dataframe(
  dframe_I,
  directory = PSA_ip_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "grayscale",
  batch_size=t_batch,
  seed= 1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  Mask_generator=mask_datagen.flow_from_dataframe(
  dframe_M,
  directory = mask_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "grayscale",
  batch_size=t_batch,
  seed=1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  train_generator = zip(zip(image_generator,pre_segment_generator), Mask_generator)
  return train_generator


def generate_valid_dataframe_cascade(dframe_val_I,dframe_val_M, img_path, mask_path, PSA_ip_path, v_batch = 8):

  Val_image_generator=val_datagen.flow_from_dataframe(
  dframe_val_I,
  directory = img_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "rgb",
  batch_size=v_batch,
  seed= 1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )



  Val_pre_segment_generator=val_segmenter_datagen.flow_from_dataframe(
  dframe_val_I,
  directory = PSA_ip_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "grayscale",
  batch_size=v_batch,
  seed= 1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  Val_Mask_generator=val_mask_datagen.flow_from_dataframe(
  dframe_val_M,
  directory = mask_path,
  x_col = "Path",
  y_col = None,
  target_size = (512, 512),
  color_mode = "grayscale",
  batch_size=v_batch,
  seed=1,
  shuffle=True,
  class_mode=None,
  interpolation = "lanczos"
  )

  val_generator = zip(zip(Val_image_generator, Val_pre_segment_generator), Val_Mask_generator)
  return val_generator

