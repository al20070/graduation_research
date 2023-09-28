import numpy as np
import keras
from keras.models import *
from keras.layers import *

def unet_3d(pretrained_weights=None, input_size=(5, 256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv3D(64, (1, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (1, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)

    conv2 = Conv3D(128, (1, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (1, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)

    conv3 = Conv3D(256, (1, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (1, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)

    conv4 = Conv3D(512, (1, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(512, (1, 3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(drop4)

    conv5 = Conv3D(1024, (1, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(1024, (1, 3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512, (1, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(1, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=4)
    conv6 = Conv3D(512, (1, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv3D(512, (1, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv3D(256, (1, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(1, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=4)
    conv7 = Conv3D(256, (1, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv3D(256, (1, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3D(128, (1, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(1, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = Conv3D(128, (1, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv3D(128, (1, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv3D(64, (1, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(1, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = Conv3D(64, (1, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv3D(64, (1, 3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv3D(2, (1, 3, 3), activation='relu', padding='same')(conv9)
    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(learning_rate=5e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
