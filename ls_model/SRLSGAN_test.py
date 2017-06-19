
# coding: utf-8

# In[1]:
import tensorflow as tf
import numpy as np
import os
import cv2
import glob
from os.path import exists
from sklearn import feature_extraction
from tqdm import trange
import pickle
# In[2]:


img_size = (96, 96)


# In[3]:


def weight_variable(shape,initializer = tf.contrib.layers.xavier_initializer()):
    return tf.get_variable("weight",shape = shape,initializer = initializer)
def bias_variable(shape, initializer = tf.contrib.layers.xavier_initializer()):
    return tf.get_variable("bias",shape = shape,initializer = initializer)
def cov2_layer(h, filters ,kernel_size, strides, padding='SAME'):
    W = weight_variable([kernel_size[0], kernel_size[1], h.get_shape().as_list()[3], filters],tf.contrib.layers.xavier_initializer_conv2d())
    b = bias_variable([filters])
    a = tf.nn.conv2d(h, W, strides=[1, strides[0], strides[1], 1], padding=padding) + b
    return a
def pRelu(x):
    alphas = tf.get_variable('alpha', x.get_shape()[1:],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5
    return pos + neg
def residual_block(h):
    with tf.variable_scope("Conv0"):
        h1 = cov2_layer(h, 64 , [3,3], [1,1])
        h1 = tf.layers.batch_normalization(h1)
        h1 = pRelu(h1)
    with tf.variable_scope("Conv1"):
        h1 = cov2_layer(h1, 64 , [3,3], [1,1])
        h1 = tf.layers.batch_normalization(h1)
        h1 = pRelu(h1)
    return h + h1


# In[4]:


def generator(x):
    with tf.variable_scope("Conv0"):
        h = cov2_layer(x, 64, [9,9], [1,1])
        h = pRelu(h)
    h1 = h
    for i in range(5):
        with tf.variable_scope("Residual_block"+str(i)):
            h = residual_block(h)
    with tf.variable_scope("Conv1"):
        h = cov2_layer(h, 64, [3,3], [1,1]) 
        h = tf.layers.batch_normalization(h)
    h = h + h1
    with tf.variable_scope("Up_Conv0"):
        h = cov2_layer(h, 256, [3,3], [1,1])
        feature_shape = h.get_shape().as_list()
        h = tf.image.resize_nearest_neighbor(h, [feature_shape[1] * 2, feature_shape[2] * 2])
        tf.reshape(h, [tf.shape(h)[0],feature_shape[1] * 2, feature_shape[2] * 2,feature_shape[3]])
        h = pRelu(h)
    with tf.variable_scope("Up_Conv1"):
        h = cov2_layer(h, 256, [3,3], [1,1])
        feature_shape = h.get_shape().as_list()
        h = tf.image.resize_nearest_neighbor(h, [feature_shape[1] * 2, feature_shape[2] * 2])
        tf.reshape(h, [tf.shape(h)[0],feature_shape[1] * 2, feature_shape[2] * 2,feature_shape[3]])
        h = pRelu(h)
    with tf.variable_scope("out_Conv"):
        sr_image = cov2_layer(h, 3, [9,9], [1,1])
    return sr_image


# In[5]:


def discriminator(x):
     
    with tf.variable_scope("Conv0"):
        h = cov2_layer(x, 64, [3,3], [1,1])
        h = tf.contrib.keras.layers.LeakyReLU()(h)
    
    with tf.variable_scope("Conv1"):
        h = cov2_layer(h, 64, [3,3], [2,2])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
     
    with tf.variable_scope("Conv2"):
        h = cov2_layer(h, 128, [3,3], [1,1])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
    
    with tf.variable_scope("Conv3"):
        h = cov2_layer(h, 128, [3,3], [2,2])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
     
    with tf.variable_scope("Conv4"):
        h = cov2_layer(h, 256, [3,3], [1,1])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
    
    with tf.variable_scope("Conv5"):
        h = cov2_layer(h, 256, [3,3], [2,2])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
     
    with tf.variable_scope("Conv6"):
        h = cov2_layer(h, 512, [3,3], [1,1])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
    
    with tf.variable_scope("Conv7"):
        h = cov2_layer(h, 512, [3,3], [2,2])
        h = tf.layers.batch_normalization(h)
        h = tf.contrib.keras.layers.LeakyReLU()(h)
    h = tf.contrib.layers.flatten(h)
    with tf.variable_scope("FC0"):
        W = weight_variable([h.get_shape().as_list()[1], 1024])
        b = bias_variable([1024])
        a = tf.matmul(h, W) + b
        h = tf.contrib.keras.layers.LeakyReLU()(a)
    with tf.variable_scope("FC1"):
        W = weight_variable([1024, 1])
        b = bias_variable([1])
        logit_y = tf.matmul(h, W) + b
        y = tf.nn.sigmoid(logit_y)
    return y,logit_y


# In[6]:


with tf.variable_scope("Input"):
    image_input = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
    test_image_input = tf.placeholder(tf.float32, [None, img_size[0]//4, img_size[1]//4, 3])
    downscaled_image = tf.image.resize_nearest_neighbor(image_input,[img_size[0]//4, img_size[1]//4])
    tf.reshape(downscaled_image, [tf.shape(image_input)[0], img_size[0]//4, img_size[1]//4, 3])
with tf.variable_scope("generator"):
    sr_image = generator(downscaled_image)
with tf.variable_scope("generator",reuse = True):
    test_sr_image = generator(test_image_input)
with tf.variable_scope("discriminator"):
    fake_y, fake_logit_y = discriminator(sr_image)
with tf.variable_scope("discriminator",reuse = True):
    true_y, true_logit_y = discriminator(image_input)


"""
# In[7]:


content_loss = tf.losses.mean_squared_error(labels = image_input,predictions = sr_image)
discrininator_loss = 0.5 * tf.losses.mean_squared_error(labels = tf.ones([tf.shape(image_input)[0],1]),predictions = true_y)+\
0.5 * tf.losses.mean_squared_error(labels = tf.zeros([tf.shape(image_input)[0],1]),predictions = fake_y)

adversarial_loss = 0.5 * tf.losses.mean_squared_error(labels = tf.ones([tf.shape(image_input)[0],1]),predictions = fake_y)

prceptual_loss =  content_loss + 1e-2 * adversarial_loss

generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
train_generator = tf.train.AdamOptimizer(1e-5).minimize(prceptual_loss, var_list = generator_vars)
generator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,"generator")

discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")
train_discriminator = tf.train.AdamOptimizer(1e-5).minimize(1e-2*discrininator_loss, var_list = discriminator_vars)
discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,"discriminator")
"""


# In[ ]:



def reconstruct_from_patches_2d(sampled_patches,image_shape = (None,None,None),strides = (None,None)):
    ah = int(image_shape[0] /strides[0] + 0.5) * strides[0]
    aw = int(image_shape[1] /strides[1] + 0.5) * strides[1]
    channel = image_shape[2]
    sh,sw = strides[0], strides[1]
    bh,bw = sampled_patches.shape[1], sampled_patches.shape[2]
    sz = sampled_patches.itemsize
    
    image = np.zeros((ah,aw,channel)).astype(np.float32)
    sum_mask = np.zeros((ah,aw)).astype(np.float32)
    mask = np.ones((bh,bw)).astype(np.float32)
    
    shape = ((ah-bh)//sh+1,(aw-bw)//sw+1,bh,bw)
    st = sz * np.array([aw*sh,sw,aw,1]) 
    for row_blocks in np.lib.stride_tricks.as_strided(sum_mask, shape=shape, strides=st):
        for i in range(len(row_blocks)):
            row_blocks[i] += mask
    for j in range(image_shape[2]):
        k = 0
        image_layer = np.zeros((ah,aw)).astype(np.float32)
        for row_blocks in np.lib.stride_tricks.as_strided(image_layer, shape=shape, strides=st):
            for i in range(len(row_blocks)):
                #bool_data = row_blocks[i] > sampled_patches[k,:,:,j]
                #row_blocks[i] = row_blocks[i]*bool_data + sampled_patches[k,:,:,j]*(1-bool_data)
                row_blocks[i] += sampled_patches[k,:,:,j]
                k = k+1
        image_layer /= sum_mask
        image[:,:,j] = image_layer
    return image[:image_shape[0],:image_shape[1],:]

minibatch_size = 200
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,"./ls_model.ckpt")
print("model restored")

image_out_dir = './super_resolution/'
if not exists(image_out_dir):
    os.makedirs(image_out_dir)
for filename in glob.glob('./low_resolution/*.jpg'):
    im = cv2.imread(filename)
    im_patches = feature_extraction.image.extract_patches_2d(im, (img_size[0]//4,img_size[1]//4))
    im_patches = im_patches / 255.*2 -1
    upsampled_patches = []
    for i in trange(int(round(im_patches.shape[0]/minibatch_size + 0.5))):
        minibatch_upsampled_patches = sess.run(test_sr_image,{test_image_input : im_patches[i*minibatch_size : (i+1)*minibatch_size]})
        upsampled_patches += minibatch_upsampled_patches.tolist()
    upsampled_patches = np.clip(np.array(upsampled_patches),-1,1).astype(np.float32)
    upsampled_im = reconstruct_from_patches_2d(upsampled_patches , (im.shape[0] * 4,im.shape[1] *4, 3) , (4,4))
    upsampled_im = ((upsampled_im+1)/2*255+0.5).astype(np.uint8)
    new_file_name = image_out_dir + filename[17:]
    cv2.imwrite(new_file_name,upsampled_im,[cv2.IMWRITE_JPEG_QUALITY, 100])
    print("wrire {0}".format(new_file_name))
sess.close()

# In[ ]:




