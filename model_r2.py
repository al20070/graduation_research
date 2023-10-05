import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def rec_res_block(input_layer, num_filters, kernel_size=(3, 3)):
    x = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(input_layer)
    x = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(x)
    x = Add()([x, input_layer])
    return x

def R2UNet(pretrained_weights=None, input_size=(256, 256, 1), num_classes=1):
    inputs = Input(input_size)
    
    print("1")
    # エンコーダ部分
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = rec_res_block(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = rec_res_block(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = rec_res_block(conv3, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    print("2")
    # デコーダ部分
    up4 = UpSampling2D(size=(2, 2))(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    merge4 = concatenate([conv4, conv3], axis=3)
    conv4 = rec_res_block(merge4, 256)
    
    up5 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    merge5 = concatenate([conv5, conv2], axis=3)
    conv5 = rec_res_block(merge5, 128)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    merge6 = concatenate([conv6, conv1], axis=3)
    conv6 = rec_res_block(merge6, 64)
    
    conv7 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv6)

    print("3")
    model = Model(inputs = inputs, outputs = conv7)

    model.compile(optimizer = Adam(learning_rate = 5e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
    