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
Python 3, OpenCV-Python, Numpy, Tqdm, Tensorflow, Picke  
  
for testing  
Scipy, Glob, Sklearn    
  
# Training  
Pickle dump batch of the encoded images from LSUN.  
Then save under the data folder, and simply type "python SRLSGAN.py".  
The loss has be modified to ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20l%5E%7B%5Ctext%7BSR%7D%7D%20%3D%20l_%7B%5Ctext%7BX%7D%7D%5E%7B%5Ctext%7BSR%7D%7D%20&plus;%2010%5E%7B-2%7Dl_%7B%5Ctext%7BGAN%7D%7D%5E%7B%5Ctext%7BSR%7D%7D) for generator, where ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20l_%7B%5Ctext%7BGAN%7D%7D%5E%7B%5Ctext%7BSR%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Cmathbb%7BE%7D_%7B%5Cmathbf%7Bz%7D%20%5Csim%20p_%5Cmathbf%7Bz%7D%28%5Cmathbf%7Bz%7D%29%7D%5Cleft%5B%20%28D%28G%28%5Cmathbf%7Bz%7D%29%29-1%29%5E2%5Cright%5D)  
Also for discriminator ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20l_%7B%5Ctext%7BDIS%7D%7D%5E%7B%5Ctext%7BSR%7D%7D%20%3D%2010%5E%7B-2%7D%5Cleft%28%20%5Cfrac%7B1%7D%7B2%7D%5Cmathbb%7BE%7D_%7B%5Cmathbf%7Bx%7D%20%5Csim%20p_%5Ctext%7Bdata%7D%28%5Cmathbf%7Bx%7D%29%7D%5Cleft%5B%20%28D%28%5Cmathbf%7Bx%7D%29-1%29%5E2%5Cright%5D&plus;%20%5Cfrac%7B1%7D%7B2%7D%5Cmathbb%7BE%7D_%7B%5Cmathbf%7Bz%7D%20%5Csim%20p_%5Cmathbf%7Bz%7D%28%5Cmathbf%7Bz%7D%29%7D%5Cleft%5B%20%28D%28G%28%5Cmathbf%7Bz%7D%29%29%29%5E2%5Cright%5D%5Cright%20%29).    

# Testing
Cd to ls_model folder, and type "python SRLSGAN_test.py".

# Measurement  
PSNR, Peak Signal to Noise Ratio  
SSIM, Structural Similarity Metric  
https://github.com/aizvorski/video-quality
