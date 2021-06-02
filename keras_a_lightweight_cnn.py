#!/usr/bin/env python
# coding: utf-8

# In[147]:


# Libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import shuffle

import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools

from keras import layers
from keras import regularizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.np_utils import to_categorical
import tensorflow.keras.optimizers as Optimizer


# In[80]:


# Load data
train_dir = r'F:\dataset\DiseaseGrading\Original_Images\Train'
test_dir = r'F:\dataset\DiseaseGrading\Original_Images\Test'
train_cate = pd.read_csv(r'F:\dataset\DiseaseGrading\Groundtruths\Train_Labels.csv')
test_cate = pd.read_csv(r'F:\dataset\DiseaseGrading\Groundtruths\Test_Labels.csv')


# In[81]:


train_cate.head()


# In[102]:


np.array(train_cate[train_cate['Image_name']=='IDRiD_001']['Retinopathy grade'])


# In[37]:


test_cate.head()


# In[166]:


# Get data and labels
train_data = []
train_labels = []

def get_images(image_dir, labels_dir):
    
    for image_file in os.listdir(image_dir):
        image = cv2.imread(image_dir+r'/'+image_file)
        image = cv2.resize(image,(227,227))
        train_data.append(image)
        labels = pd.read_csv(labels_dir)
        label = list(labels[labels['Image_name'] + '.jpg' ==  image_file]['Retinopathy grade'])
        train_labels.append(label)
    
    return shuffle(train_data,train_labels,random_state=7)


# In[167]:


train_data, train_labels = get_images(r'F:\dataset\DiseaseGrading\Original_Images\Train', r'F:\dataset\DiseaseGrading\Groundtruths\Train_Labels.csv')


# In[168]:


train_data = np.array(train_data)
train_labels = np.array(train_labels)


# In[169]:


train_data.shape


# In[170]:


train_labels.shape


# In[171]:


train_labels.reshape(413)


# In[137]:


# for image_file in os.listdir(r'F:\dataset\DiseaseGrading\Original_Images\Train'):
#     print(image_file)


# In[161]:


# train_labels = to_categorical(train_labels)


# In[162]:


# train_labels


# In[144]:


# Build cnn model
model = keras.Sequential([
tf.keras.Input(shape = (227, 227, 3)),
layers.Conv2D (64, (5,5), strides=(1, 1), padding='same', kernel_regularizer = regularizers.l2(0.001), 
               bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope= 1),
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

layers.Conv2D (64, (5,5), strides=(1, 1), padding='same', kernel_regularizer = regularizers.l2(0.001), 
               bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1),
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=(2, 2)),

layers.Conv2D (128, (5,5), strides=(1, 1), padding='same', kernel_regularizer = regularizers.l2(0.001), 
               bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1),
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=(2, 2)),

layers.Conv2D (256, (3,3), strides=(1, 1), padding='same', kernel_regularizer = regularizers.l2(0.001), 
               bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1),
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

layers.Conv2D (512, (3,3), strides=(1, 1), padding='same', kernel_regularizer = regularizers.l2(0.001), 
               bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1), #negative slope 1 is good
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

layers.Conv2D (1024, (1,1), strides=(1, 1), padding='same', kernel_regularizer = regularizers.l2(0.001), 
               bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1),
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
layers.MaxPooling2D(pool_size=(3, 3), padding="same"),
layers.Reshape((1, 4096)),
layers.Dense(1024, kernel_regularizer = regularizers.l2(0.001), bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1),
layers.Dropout(0.5),
layers.Dense(32, name="my_intermediate_layer", kernel_regularizer = regularizers.l2(0.001), 
             bias_regularizer = regularizers.l2(0.001)),
layers.ReLU(negative_slope = 1),
layers.Dense(5, activation='softmax', name="my_last_layer"),
])


# In[145]:


model.summary()


# In[187]:


model.compile(optimizer=Optimizer.Adam(lr=0.00005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[188]:


# Train
trained = model.fit(train_data,train_labels,epochs=30,validation_split=0.10)


# In[189]:


# SVM
model_feat = Model(inputs=model.input,outputs=model.get_layer('my_intermediate_layer').output)
feat_train = model_feat.predict(train_data)
print(feat_train.shape)


# In[190]:


feat_train = feat_train.reshape(413,32)


# In[191]:


from sklearn.svm import SVC

svm = SVC(kernel='rbf')
svm.fit(feat_train,train_labels)
svm.score(feat_train,train_labels)

