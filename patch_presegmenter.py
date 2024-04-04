
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
#from tensorflow import keras
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import scipy.misc as sc
import cv2
import random
import glob
import shutil
import data_preprocessing as dp

def get_patch_crops(im,mask):
    flipped_mask = mask.astype(np.uint8)
    flipped_img = im

    flipped_mask = cv2.copyMakeBorder(flipped_mask, 512, 512, 512, 0, cv2.BORDER_CONSTANT, None, value = 0)
    flipped_img = cv2.copyMakeBorder(flipped_img, 512, 512, 512, 0, cv2.BORDER_CONSTANT, None, value = 0)

    _, breast_mask = cv2.threshold(flipped_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cnts,hi = cv2.findContours(breast_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    k = 0
    crop_img_list = []
    crop_mask_list = []
    for i in cnts:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x = cx-512
            y = cy-512
            h = 1024
            w = 1024
            print(y,cy+512)
            print(x,cx+512)

            crop_img = flipped_img[y:cy+512,x:cx+512]
            crop_mask = flipped_mask[y:cy+512,x:cx+512]
            k = k + 1
            print(k)
            crop_img_list.append(crop_img)
            crop_mask_list.append(crop_mask)
    return crop_img_list,crop_mask_list

def prepare_training_patches(I_list, img_path, mask_path):
  train_data = []
  train_mask = []
  labels_list = []
  for i in I_list:
    j = i.rsplit(".",1)[0]
    mask = Image.open(img_path+str(i))
    mask = np.array(mask)
    img = Image.open(mask_path+str(i))
    img = np.array(img)
    norm_img = dp.normalize_image_clahe(img)
    print(norm_img.shape)
    print(mask.shape)
    crop_img_list,crop_mask_list = get_patch_crops(norm_img,mask,j)
    for i in range(len(crop_img_list)):
      h,w,c = crop_img_list[i].shape
      crop_img = crop_img_list[i]
      crop_mask = crop_mask_list[i]
      if(h<1024):
          crop_img = cv2.copyMakeBorder(crop_img, 0,1024 - h, 0, 0,cv2.BORDER_CONSTANT, value = 0)
          crop_mask = cv2.copyMakeBorder(crop_mask, 0,1024 - h, 0, 0,cv2.BORDER_CONSTANT, value = 0)
          h = 1024
      if(w<1024):
          crop_img = cv2.copyMakeBorder(crop_img, 0,0, 0, 1024 - w ,cv2.BORDER_CONSTANT, value = 0)
          crop_mask = cv2.copyMakeBorder(crop_mask, 0,0, 0, 1024 - w ,cv2.BORDER_CONSTANT, value = 0)
          w = 1024

      crop_img = cv2.resize(crop_img,(256,256))
      crop_mask = cv2.resize(crop_mask,(256,256))

      train_data.append(crop_img)
      train_mask.append(crop_mask/255.0)
  return train_data, train_mask


def prepare_test_patches(I_list,img_path,mask_path):
  train_data = []
  train_mask = []
  labels_list = []
  for i in I_list:
    j = i.rsplit(".",1)[0]
    mask = Image.open(img_path+str(i))
    mask = np.array(mask)
    img = Image.open(mask_path+str(i))
    img = np.array(img)
    norm_img = dp.normalize_image_clahe(img)
    print(norm_img.shape)
    print(mask.shape)
    crop_img_list,crop_mask_list = get_patch_crops(norm_img,mask,j)
    for i in range(len(crop_img_list)):
      h,w,c = crop_img_list[i].shape
      crop_img = crop_img_list[i]
      crop_mask = crop_mask_list[i]
      if(h<1024):
          crop_img = cv2.copyMakeBorder(crop_img, 0,1024 - h, 0, 0,cv2.BORDER_CONSTANT, value = 0)
          crop_mask = cv2.copyMakeBorder(crop_mask, 0,1024 - h, 0, 0,cv2.BORDER_CONSTANT, value = 0)
          h = 1024
      if(w<1024):
          crop_img = cv2.copyMakeBorder(crop_img, 0,0, 0, 1024 - w ,cv2.BORDER_CONSTANT, value = 0)
          crop_mask = cv2.copyMakeBorder(crop_mask, 0,0, 0, 1024 - w ,cv2.BORDER_CONSTANT, value = 0)
          w = 1024

      crop_img = cv2.resize(crop_img,(256,256))
      crop_mask = cv2.resize(crop_mask,(256,256))
      train_data.append(crop_img)
      train_mask.append(crop_mask/255.0)
  return train_data, train_mask


def build_saliency_map(predicted_patches,h,w,stride):
    patch_size = predicted_patches.shape[1]  # patches assumed to be square
    rows = h
    cols = w
    reconim = np.zeros((rows, cols))
    patch_num = 0

    for i in range(0, h - patch_size + 1 , stride):
        for j in range(0, w - patch_size + 1 , stride):
            reconim[i:i + patch_size, j:patch_size + j] = np.max(reconim[i:i + patch_size, j:patch_size + j],predicted_patches[patch_num])
            patch_num += 1

    reconstructedim = reconim
    return reconstructedim



def get_test_saliency_map(patch_seg_model, test_data,test_path,test_mask_path,stride):
  for i in test_data:
    g = cv2.imread(str(test_mask_path)+str(i),0)
    g = g/255.0

    I = cv2.imread(str(test_path)+str(i),0)

    h,w = I.shape
    print(h,w)
    I = cv2.copyMakeBorder(I, 512, 512, 512, 512, cv2.BORDER_CONSTANT, None, value = 0.0)
    g = cv2.copyMakeBorder(g, 512, 512, 512, 512, cv2.BORDER_CONSTANT, None, value = 0.0)

    h = h + 1024
    w = w + 1024

    In = dp.normalize_image_clahe(I)
    #Extract patches from test image
    image = tf.expand_dims(np.array(In), 0)
    patches = tf.image.extract_patches(images=image,
                            sizes=[1, 1024, 1024, 1],
                            strides=[1, 512, 512, 1],
                            rates=[1, 1, 1, 1],
                            padding='VALID')
    patches = tf.reshape(patches, (patches.shape[1]*patches.shape[2],1024,1024,3))
    predicted_patches = []

    for p in patches:
        p = np.asarray(p)
        p = cv2.resize(p,(256,256))
        p = np.reshape(p,(1,)+p.shape)
        pre = patch_seg_model.predict(p)[0]
        pre = cv2.resize(pre, (256,256))
        pre = cv2.resize(pre,(1024,1024),interpolation = cv2.INTER_LINEAR)
        predicted_patches.append(pre)
        
    predicted_patches = np.asarray(predicted_patches)
    reconstructed_pre = build_saliency_map(predicted_patches,h,w,stride)
    reconstructed_pre_o = reconstructed_pre[512:h-512,512:w-512]
    
  return reconstructed_pre_o

