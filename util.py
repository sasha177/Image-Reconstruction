# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:02:26 2018

@author: ZMJ
"""
from PIL import Image
import numpy as np

#borrowed func, adapted for clarity
def pad_img(img: Image):
  
  s1, s2 = img.size
  diag = int(np.ceil(np.hypot(s1,s2)))

  # square with new side length diag
  img_pad = Image.new("L",(diag,diag))
  
  # paste img in top left corner (with padding)
  start_x = int(round((diag-s1)/2))
  start_y = int(round((diag-s2)/2))
  img_pad.paste(img,(start_x,start_y))

  return img_pad, start_x, start_y

#borrowed func, annotated for clarity
#compactly creates a sinogram from an image
def projection(img: Image, thetas):
  
  num_thetas = len(thetas)

  # sino will have same width as img
  sinogram = np.zeros((img.size[1], num_thetas))

  for i, theta in enumerate(thetas):
    rot_img = img.rotate(theta, resample=Image.BICUBIC)

    # sum rotated image along cols
    # save result in sinogram row
    sinogram[:,i] = np.sum(rot_img,axis=0)

  return sinogram

def project_array(img_array, thetas):
  img = Image.fromarray(img_array)
  return projection(img, thetas)