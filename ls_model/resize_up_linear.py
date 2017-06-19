import os
import cv2
import glob
from os.path import exists

image_out_dir = './image_resized_linear/'
if not exists(image_out_dir):
    os.makedirs(image_out_dir)
for filename in glob.glob('./low_resolution/*.jpg'): #assuming gif
    im = cv2.imread(filename)
    new_file_name = image_out_dir + filename[17:]
    im = cv2.resize(im,(im.shape[1]*4,im.shape[0]*4))
    cv2.imwrite(new_file_name,im,[cv2.IMWRITE_JPEG_QUALITY, 100])

