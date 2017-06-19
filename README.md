# CCBDA-project
Paper implementation for Super Resolution using a Generative Adversarial Network with Least Square modification  
https://arxiv.org/abs/1609.04802  
https://arxiv.org/abs/1611.04076  

# Dataset
The images used from training in the project are first 190000 images of Bedroom part of Large-scale Scene Understanding Challenge (LSUN) dataset.  
http://lsun.cs.princeton.edu/2015.html  
The images for testing are random sampled 20 images of training part of Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500).  
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html  

# Prerequirement
for training
Python 3  
OpenCV-Python  
Numpy  
Tqdm  
Tensorflow
Picke  
  
for testing
Scipy  
Glob 
Sklearn  

# Training
Pickle dump batch of the encoded images from LSUN.  
Then save under the data folder, and simply type "python SRLSGAN.py".

# Testing
Cd to ls_model folder, and type "python SRLSGAN_test.py".

# Measurement  
PSNR, Peak Signal to Noise Ratio  
SSIM, Structural Similarity Metric  
https://github.com/aizvorski/video-quality
