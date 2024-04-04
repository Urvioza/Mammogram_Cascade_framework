import numpy as np
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred,smooth = 0.0001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def finddice_score(y_pred, y_true, smooth = 0.0001):
  intersection = np.sum(np.abs(y_pred * y_true))
  mask_sum = np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred))
  union = mask_sum  - intersection
  smooth = 0.0001
  dice = (2 * intersection + smooth)/(mask_sum + smooth)

  return dice

def dice_coef_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))

def false_positive_rate(p,g):
   fp = np.sum(p == 1) - np.sum(p*g)
   return (fp/(np.sum(g==0)))