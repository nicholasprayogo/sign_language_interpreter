#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:21:47 2018

@author: nick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('sign_mnist_train.csv')
df=pd.DataFrame(df)

def data_insights():
    print(df)
    print(df.head())
    print(df.shape)
    print(len(df.columns))
    print(df['label'].nunique())

#Visualize data
def visualize():
    n=0
    for i in range(25):
        image=np.array(df.iloc[n,1:len(df.columns)])
        A = df.iloc[n,0]
        B = np.reshape(image, (28, 28))
        plt.subplot(5,5,n+1)
        plt.imshow(B, 'gray', vmin=0, vmax=255)
        plt.title(chr(97+A))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        n+=1
    plt.tight_layout()
    plt.show()
    #plt.savefig('sign_data.png')

def split():
    # n=df.shape[0]
    #     # # count=0
    #     # x_= np.zeros((n, 28, 28, 1), dtype='float64')
    #     # # for i in
    x=np.array(df.iloc[:,1:len(df.columns)])
    #Change x to a shape that can be processed by the NN layers
    #Input data needs [None, 28, 28, 1] where on each iteration last input would be greyscale value
    #We want e.g. [0, 2, 4, 255]

    #IMPORTANT CRUCIAL TO ADD THAT 1 AT THE END TO MATCH WITH PLACEHOLDER DIMENSIONS
    x= np.reshape(x, (-1, 28, 28, 1))
    print(x)
    y=np.array(df.iloc[:,0])
    x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=15)
    #make sure random state is on
    return(x_train, x_test, y_train, y_test)

x_train, x_test, y_train, y_test=split()

print(x_train)
print(y_train)

#one hot encode Y cus we need size: (batch_size, 24) not (y.shape[0])
from sklearn.preprocessing import OneHotEncoder
y_train= y_train.reshape(-1,1)
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test= y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test).toarray()

print(y_train)

#import tflearn methods
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.models.dnn import DNN
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.estimator import regression
import tensorflow as tf



# # Make sure the data is normalized
# img_prep = ImagePreprocessing()
# img_prep.add_featurewise_stdnorm()
#
# # Create extra synthetic training data by flipping, rotating and blurring the
# # images on our data set.
# img_aug = ImageAugmentation()
# img_aug.add_random_flip_leftright()
# img_aug.add_random_rotation(max_angle=25.)
# img_aug.add_random_blur(sigma_max=3.)

tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True)

def build_model():
    #config.gpu_options.per_process_gpu_memory_fraction = 0.01
    with tf.Session(config=config) as sess:
        #make sure input_data 's placeholder shape corresponds to model shape
        network = input_data(shape=[None, 28,28,1])
        #shape: None (Placeholder for number of training sets in batch), 28 by 28 pixels, 1 channel (greyscale)
        #now network becomes a 4D tensor with dimensions [batch, height, width, in_channels]
        network = conv_2d(network, 16, [5,5], activation='relu')
        #now network becomes a 4D tensor with dimensions [batch, new height, new width, n_filters]
        network = max_pool_2d(network, [2,2])
        #filter size [2,2], stride is implied: 2
        network = conv_2d(network, 64, 3, activation='relu')
        #use relu to account for non-linearity
        network = max_pool_2d(network, 2)
        #same as [2,2]
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 24, activation='softmax')
        network = regression(network, optimizer='sgd', loss='categorical_crossentropy',learning_rate=0.01)
        #always remember to test different optimizers, set learning rates accordingly too
        model = DNN(network, tensorboard_verbose=3, checkpoint_path='sign_language_model.ckpt')

    return(model)

def train():
    model=build_model()
    model.fit(x_train, y_train, n_epoch=10, shuffle=True, validation_set=(x_test, y_test),show_metric=True, batch_size=20)
    # feed dict { 'inputs': x_train } { 'targets': y_train }
    model.save("ckpts/sign-language-classifier.tfl")
    return(model)

visualize()
train()

print(DNN.evaluate(model, x_test, y_test, batch_size=96))
