
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import proccess2
from tqdm import tqdm,trange


# In[2]:


img_size = (96, 96)
train_data = proccess2.LSUN(["train"+str(i) for i in range(1,20)],img_size)
test_data = proccess2.LSUN(["train20"],img_size)
val_data = proccess2.LSUN(["val"],img_size)


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



# In[ ]:


minibatch_size = 25
train_loss = dict(zip(["dis","gen_x","gen_gen","gen_sr"],[0.0]*4))
val_loss = train_loss.copy()
test_loss = train_loss.copy()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("model restored")
saver.restore(sess,"./ls_model/ls_model.ckpt")
for i in tqdm(range(10)):
    for j in tqdm(range(190000 // minibatch_size)):
        batch_image = train_data.get_next_batch(minibatch_size)/255.* 2 - 1
        _,_,train_loss["dis"] = sess.run([train_discriminator, discriminator_update_ops,discrininator_loss], {image_input: batch_image})
        print("train discriminator loss : {0}".format(train_loss["dis"]))
        
        _,_,train_loss["gen_x"],train_loss["gen_gen"],train_loss["gen_sr"] = sess.run([train_generator, generator_update_ops, content_loss, adversarial_loss, prceptual_loss], {image_input: batch_image})
        print("train generator loss : {2}, content_loss : {0},  adversarial_loss : {1}".format(train_loss["gen_x"],train_loss["gen_gen"],train_loss["gen_sr"]))
    for j in tqdm(range(300 // minibatch_size)):
        batch_image = val_data.get_next_batch(minibatch_size)/255.* 2 - 1
        tmp = sess.run(discrininator_loss, {image_input: batch_image})
        val_loss["dis"] += tmp
        tmp1,tmp2,tmp3 = sess.run([content_loss, adversarial_loss, prceptual_loss], {image_input: batch_image})
        val_loss["gen_x"] += tmp1
        val_loss["gen_gen"] += tmp2
        val_loss["gen_sr"] += tmp3
    val_loss["dis"] /= 3000 // minibatch_size
    val_loss["gen_x"] /= 3000 // minibatch_size
    val_loss["gen_gen"] /= 3000 // minibatch_size
    val_loss["gen_sr"] /= 3000 // minibatch_size
    print("val discriminator loss : {3}, val generator loss : {2}, content_loss : {0},  adversarial_loss : {1}".          format(val_loss["gen_x"],val_loss["gen_gen"],val_loss["gen_sr"],val_loss["dis"]))
    print("model saved")
    saver.save(sess, "./ls_model/ls_model.ckpt")
for j in tqdm(range(10000 // minibatch_size)):
    batch_image = test_data.get_next_batch(minibatch_size)/255.* 2 - 1
    tmp = sess.run(discrininator_loss, {image_input: batch_image})
    test_loss["dis"] += tmp
    tmp1,tmp2,tmp3 = sess.run([content_loss, adversarial_loss, prceptual_loss], {image_input: batch_image})
    test_loss["gen_x"] += tmp1
    test_loss["gen_gen"] += tmp2
    test_loss["gen_sr"] += tmp3
test_loss["dis"] /= 10000 // minibatch_size
test_loss["gen_x"] /= 10000 // minibatch_size
test_loss["gen_gen"] /= 10000 // minibatch_size
test_loss["gen_sr"] /= 10000 // minibatch_size
print("test discriminator loss : {3}, test generator loss : {2}, content_loss : {0},  adversarial_loss : {1}".     format(test_loss["gen_x"],test_loss["gen_gen"],test_loss["gen_sr"],test_loss["dis"]))
sess.close()

# In[ ]:




