import ssim
import psnr
import glob
import numpy as np
import cv2
import csv
import math


header = ["File name","Low Resolution", "Leaner","Super Resoultion"]
psnr_measure = [header.copy()]
ssim_measure = [header.copy()]
for filename in glob.glob('./high_resolution/*.jpg'): #assuming gif
    hr_im = cv2.imread(filename)
    sr_im = cv2.imread('./super_resolution' + filename[17:])
    rl_im = cv2.imread('./image_resized_linear' + filename[17:])
    lr_im = cv2.imread('./image_upsampled' + filename[17:])
    hr_im_gray = cv2.cvtColor(hr_im, cv2.COLOR_BGR2GRAY)
    sr_im_gray = cv2.cvtColor(sr_im, cv2.COLOR_BGR2GRAY)
    rl_im_gray = cv2.cvtColor(rl_im, cv2.COLOR_BGR2GRAY)
    lr_im_gray = cv2.cvtColor(lr_im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("HR", hr_im_gray)
    #cv2.waitKey()
    psnr_measure_list = [filename[18:], psnr.psnr(lr_im,hr_im),psnr.psnr(rl_im,hr_im),psnr.psnr(sr_im,hr_im)]
    ssim_measure_list = [filename[18:], ssim.ssim(lr_im_gray,hr_im_gray),ssim.ssim(rl_im_gray,hr_im_gray),ssim.ssim(sr_im_gray,hr_im_gray)]
    psnr_measure.append(psnr_measure_list)
    ssim_measure.append(ssim_measure_list)
with open("psnr.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(psnr_measure)
with open("ssim.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(ssim_measure)
