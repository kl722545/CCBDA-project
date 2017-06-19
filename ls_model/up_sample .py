import os
import cv2
import glob
from os.path import exists
import numpy as np

image_out_dir = './image_upsampled/'
if not exists(image_out_dir):
    os.makedirs(image_out_dir)
for filename in glob.glob('./low_resolution/*.jpg'): #assuming gif
    im = cv2.imread(filename)
    new_file_name = image_out_dir + filename[17:]
    im = np.repeat(np.repeat(im,4,axis = 0),4,axis = 1)
    cv2.imwrite(new_file_name,im,[cv2.IMWRITE_JPEG_QUALITY, 100])

