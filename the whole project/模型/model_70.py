#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 08:48:46 2018

@author: biototem
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import argparse
import keras
import os
from pathlib import Path
import sys
import glob
#import deep_learning_utils
from keras.models import *
from keras.optimizers import *
import metrics
from keras import regularizers
from keras.layers import advanced_activations
from keras import initializers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau

train_data_dir = './breast70/train'
validation_data_dir = './breast70/test'
img_width, img_height = 70,70

train_datagen = ImageDataGenerator(
                rescale=1./255)
validation_datagen = ImageDataGenerator(
                rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size = (img_width,img_height),
                batch_size = 32,
                class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
                validation_data_dir,
                target_size = (img_height,img_width),
                batch_size = 32,
                class_mode = 'categorical')

nb_train_samples = sum([len(files) for root,dirs,files in os.walk('./breast70/train')])
nb_validation_samples = sum([len(files) for root,dirs,files in os.walk('./breast70/test')])

def get_callbacks(filepath,model,patience=4):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint('./model70/'+filepath, monitor='val_loss',mode='min', save_best_only=True)
    file_dir = './model70/log/'+model + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./model70/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) + '_'+ model +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

import time
file_path = "simple_cnn_1.h5"
callbacks_s = get_callbacks(file_path,'simple_cnn',patience=10)

from keras.layers import Conv2D,MaxPooling2D
model = Sequential()
model.add(Conv2D(50, (3, 3), input_shape=(img_width, img_height, 3), activation='relu',kernel_initializer='he_normal',name="CONV1"))
model.add(MaxPooling2D(pool_size=(2, 2),name="MP1"))

model.add(Conv2D(100, (3, 3), kernel_initializer='he_normal',activation='relu', name="CONV2"))
model.add(MaxPooling2D(pool_size=(2, 2),name="MP2"))
model.add(Flatten())
model.add(Dense(500, kernel_regularizer=regularizers.l2(0.05), activation='relu', kernel_initializer='he_normal', name="FC1"))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax',  name="FC3_output"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4), 
              metrics=['accuracy', metrics.precision, metrics.recall, metrics.fscore])

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples //32,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples//32,
        callbacks = callbacks_s,
    verbose=1)

validation_generator.class_indices