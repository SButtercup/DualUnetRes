#!/usr/bin/env python
# coding: utf-8

# In[4]:


#libraries
"""
@author: Shelia Rahman Tuly
"""

import tensorflow as tf

import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from tqdm import notebook, tnrange,tqdm


from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img


# In[5]:


img_height=512
img_width=512
img_channels=3

#train_data
ids = next(os.walk("C:/ZMyFiles/research/train_GBM/train_GBM/original"))[2] # list of names all images in the given path
print("No. of images = ", len(ids))
#test data
test_ids = next(os.walk("C:/ZMyFiles/research/train_GBM/train_GBM/original_test"))[2] # list of names all images in the given path
print("No. of images = ", len(test_ids))



X = np.zeros((len(ids), img_height, img_width, 1), dtype=np.float32)
Y = np.zeros((len(ids), img_height, img_width, 1), dtype=np.float32)
Z = np.zeros((len(test_ids), img_height, img_width, 1), dtype=np.float32)
T=  np.zeros((len(test_ids), img_height, img_width, 1), dtype=np.float32)


# In[6]:


for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    # Load images
    img = load_img("C:/ZMyFiles/research/train_GBM/train_GBM/original/"+id_, color_mode = "grayscale")
    x_img = img_to_array(img)
    x_img = resize(x_img, (512, 512, 1), mode = 'constant', preserve_range = True)
    #x_img = resize(x_img, (256, 256, 1), mode = 'constant', preserve_range = True)
     # Load masks
    #mask_=np.zeros((img_height,img_width,1),dtype=np.bool)
    mask = img_to_array(load_img("C:/ZMyFiles/research/train_GBM/train_GBM/masks/"+id_, color_mode = "grayscale"))
    mask = resize(mask, (512, 512, 1), mode = 'constant', preserve_range = True)
    #mask = resize(mask, (256, 256, 1), mode = 'constant', preserve_range = True)
    
    #mask_=np.minimum(mask_,mask)
    # Save images
    X[n] = x_img/255.0
    Y[n] = mask/255.0


# In[7]:


file_list=[]
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    #load test images
    img_test = load_img("C:/ZMyFiles/research/train_GBM/train_GBM/original_test/"+id_, color_mode = "grayscale")
    x_test_img = img_to_array(img_test)
    x_test_img = resize(x_test_img, (512, 512, 1), mode = 'constant', preserve_range = True)
    #x_test_img = resize(x_test_img, (256, 256, 1), mode = 'constant', preserve_range = True)
    file_list.append(id_)
    
    mask_test = img_to_array(load_img("C:/ZMyFiles/research/train_GBM/train_GBM/masks_test/"+id_, color_mode = "grayscale"))
    mask_test = resize(mask, (512, 512, 1), mode = 'constant', preserve_range = True)
    
    #print(id_)
    #save images
    
    Z[n]=x_test_img/255.0
    T[n]=mask_test/255.0
   


# In[9]:


len(Z)


# In[8]:


x_train, x_valid, y_train, y_valid = train_test_split(X ,Y, test_size=0.2, random_state=42)


# In[9]:


def resnext_block(x , filter_size, size, stride=1):
    conv1 = tf.keras.layers.Conv2D(size, (filter_size,filter_size),strides=stride , padding='same')(x)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    conv1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
    
    filter_size2 = filter_size*2
    conv2 = tf.keras.layers.Conv2D(size, (filter_size2,filter_size2),strides=stride , padding='same')(x)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    conv2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
    
    
    filter_size3 = filter_size*3
    conv3 = tf.keras.layers.Conv2D(size, (filter_size2,filter_size2),strides=stride , padding='same')(x)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    conv3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)
    print("before concatenation :", tf.shape(conv3))
    concate1=tf.keras.layers.concatenate([conv1,conv2])
    concate2=tf.keras.layers.concatenate([concate1,conv3])
    print("after concatenation :", tf.shape(concate2))
    return concate2
    


# In[10]:


def the_unet(inputs):
    #s=tf.keras.layers.add.Lambda(lambda x:x/255)(inputs)
    c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1=tf.keras.layers.Dropout(0.1)(c1)
    c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
    p1= resnext_block(c1,filter_size=1, size=16, stride=1)
    #p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

    
    c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2=tf.keras.layers.Dropout(0.1)(c2)
    c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
    p2= resnext_block(c2,filter_size=1, size=32, stride=1)
    #p2=tf.keras.layers.MaxPooling2D((2,2))(c2)
    
    c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3=tf.keras.layers.Dropout(0.2)(c3)
    c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
    p3= resnext_block(c3,filter_size=1, size=64, stride=1)
    #p3=tf.keras.layers.MaxPooling2D((2,2))(c3)
    
    c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4=tf.keras.layers.Dropout(0.2)(c4)
    c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
    p4= resnext_block(c4,filter_size=1, size=128, stride=1)
    #p4=tf.keras.layers.MaxPooling2D((2,2))(c4)
    
    c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5=tf.keras.layers.Dropout(0.3)(c5)
    c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
    p5= resnext_block(c5,filter_size=1, size=256, stride=1)
    #p5=tf.keras.layers.MaxPooling2D((2,2))(c5)
    
    c6=tf.keras.layers.Conv2D(512,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6=tf.keras.layers.Dropout(0.3)(c6)
    c6=tf.keras.layers.Conv2D(512,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)
    #p6=tf.keras.layer.Maxpooling2D((2,2))(c6)
    
    #Segmentation layer
    u7=tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c6)
    u7=tf.keras.layers.concatenate([u7,c5])
    c7=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7)
    c7=tf.keras.layers.Dropout(0.2)(c7)
    c7=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)
    
    u8=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c7)
    u8=tf.keras.layers.concatenate([u8,c4])
    c8=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
    c8=tf.keras.layers.Dropout(0.2)(c8)
    c8=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)
    
    u9=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c8)
    u9=tf.keras.layers.concatenate([u9,c3])
    c9=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
    c9=tf.keras.layers.Dropout(0.1)(c9)
    c9=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)
    
    u10=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c9)
    u10=tf.keras.layers.concatenate([u10,c2])
    c10=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u10)
    c10=tf.keras.layers.Dropout(0.1)(c10)
    c10=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c10)
    
    u11=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c10)
    u11=tf.keras.layers.concatenate([u11,c1],axis=3)
    c11=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u11)
    c11=tf.keras.layers.Dropout(0.1)(c11)
    c11=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c11)
    
    #regression layer
    # v7=tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c6)
    # v7=tf.keras.layers.concatenate([v7,c5])
    # c_7=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(v7)
    # c_7=tf.keras.layers.Dropout(0.2)(c_7)
    # c_7=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c_7)
    
    v8=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
    v8=tf.keras.layers.concatenate([v8,c4])
    c_8=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(v8)
    c_8=tf.keras.layers.Dropout(0.2)(c_8)
    c_8=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c_8)
    
    v9=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c_8)
    v9=tf.keras.layers.concatenate([v9,c3])
    c_9=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(v9)
    c_9=tf.keras.layers.Dropout(0.1)(c_9)
    c_9=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c_9)
    
    v10=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c_9)
    v10=tf.keras.layers.concatenate([v10,c2])
    c_10=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(v10)
    c_10=tf.keras.layers.Dropout(0.1)(c_10)
    c_10=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c_10)
    
    v11=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c_10)
    v11=tf.keras.layers.concatenate([v11,c1],axis=3)
    c_11=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(v11)
    c_11=tf.keras.layers.Dropout(0.1)(c_11)
    c_11=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c_11)
    
    #fusion layer
    l1=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c11)
    l2=tf.keras.layers.Conv2D(1,(1,1),activation='linear')(c_11)
    cl1=tf.keras.layers.concatenate([l1,l2])
    
    #chnaged 03/15/2022 changed 256 to 128
    conl1=tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),activation='relu',kernel_initializer='he_normal',padding='same')(cl1)
    bn1=tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(conl1)
    relu1=tf.keras.layers.ReLU()(bn1)
    
    #convl1=tf.keras.layers.add([conl1,bn1,relu1])
    #changed 256 to 128
    conl2=tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),activation='relu',kernel_initializer='he_normal',padding='same')(relu1)
    bn2=tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(conl2)
    relu2=tf.keras.layers.ReLU()(bn2)
    
    #convl2=tf.keras.layers.add([conl2,bn2,relu2])
    
    # conl1=tf.keras.layers.Conv2D(1,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(cl1)
    # conl2=tf.keras.layers.Conv2D(1,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conl1)

    #changed :kernel_initializer='he_normal',
    # conl1=tf.keras.layers.Conv2D(1,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(cl1)
    # conl1=tf.keras.layers.BatchNormalization()(conl1)
    # conl1=tf.keras.layers.ReLU()(conl1)
    # conl2=tf.keras.layers.Conv2D(1,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conl1)
    # conl1=tf.keras.layers.BatchNormalization()(conl2)
    # conl1=tf.keras.layers.ReLU()(conl2)
    
    #changed 3/10/2022
    # cl1.add(tf.keras.layers.Conv2D(1,(3,3),kernel_initializer='he_normal',padding='same'))
    # cl1.add(tf.keras.layers.BatchNormalization())
    # cl1.add(tf.keras.layers.ReLU())
    # cl1.add(tf.keras.layers.Conv2D(1,(3,3),kernel_initializer='he_normal',padding='same'))
    # cl1.add(tf.keras.layers.BatchNormalization())
    # cl1.add(tf.keras.layers.ReLU())
    
    outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(relu2)    
    
    loss1=tf.keras.losses.BinaryCrossentropy()   
    loss2=tf.keras.losses.MeanSquaredError() 
    #total_loss=loss1(inputs,l1)+loss2(inputs,l2)+loss1(inputs,outputs)
    
    model=tf.keras.Model(inputs=[inputs],outputs=[l1,outputs,l2])
    model.compile(optimizer='adam',loss=[loss1,loss2,loss1],metrics=['accuracy'])
    model.summary()
    return model


inputs=tf.keras.layers.Input((img_height,img_width,1))
model = the_unet(inputs)


# In[ ]:


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000001, verbose=1),
    ModelCheckpoint('All3Categories_nuclie.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


results = model.fit(x_train, y_train, batch_size=16, epochs=1, callbacks=callbacks,validation_data=(x_valid, y_valid))


# In[28]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()


# In[57]:



model.evaluate(x_valid, y_valid,batch_size=4, verbose=1)



#preds_val = model.predict(x, verbose=1)
preds_test = model.predict(Z, verbose=1)
#model.evaluate(x_valid, y_valid,batch_size=4, verbose=1)


# In[21]:


len(preds_test)
len(preds_test[0])


# In[40]:


preds_test.shape


# In[97]:


def plot_sample(X, Y, preds, ix):
    """Function to plot the results"""
#     if ix is None:
#         ix = random.randint(0, len(X))

    has_mask = Y[ix].max() > 0

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
#     ax[0,0].imshow(X[ix, ..., 0], cmap='gray')
#     if has_mask:
#         ax[0,0].contour(Y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[0,0].set_title('Original Image with Mask')
#     ax[0,1].imshow(Y[ix].squeeze(), cmap='gray')
#     ax[0,1].set_title('Mask')
    
    ax[0,0].imshow(Z[ix].squeeze(),cmap='twilight_shifted')
    ax[0,0].set_title('Original Image for Prediction')
    
    ax[0,1].imshow(preds[0][ix].squeeze(), cmap='twilight', vmin=0, vmax=1)
    ax[0,1].set_title('Output from Segmetation Layer')
    
    ax[1,0].imshow(preds[0][ix].squeeze(), cmap='twilight',interpolation='nearest')
    ax[1,0].set_title('Output from Regression Layer')
    
    ax[1,1].imshow(preds[0][ix].squeeze(), cmap='twilight')
    ax[1,1].set_title('Nuclei Prediction (Dual U-net)')
    
    #if has_mask:
    #   ax[1,1].contour(Z[ix].squeeze(), colors='b', levels=[0.3])
    #ax[1,1].set_title('Nuclei Prediction')
    
ix = 27
out = plot_sample(X, Y, preds_test, ix)
tmp = file_list[ix].split(".")
plt.savefig(tmp[0]+"_pred.png")


# In[ ]:





# In[ ]:




