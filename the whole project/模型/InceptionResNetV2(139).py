# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:33:31 2018

@author: Biototem_1
"""

import pandas as pd
import numpy as np
import argparse
import datetime
#import GPUtil
import random
import keras
import glob
import time
import sys
import os

from keras.models import *
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.initializers import Orthogonal
from keras.utils import to_categorical
from keras.preprocessing import image
from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import multi_gpu_model

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base=load_model('InceptionResNetV2(139).h5')

top_model = Sequential()
top_model.add(base)
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax',kernel_initializer=Orthogonal()))
parallel_model = multi_gpu_model(top_model,2)
top_model.summary()

for layer in parallel_model.layers:
    layer.trainable = True

LearningRate = 0.01
decay = 0.0001
n_epochs = 20
sgd = optimizers.SGD(lr=LearningRate, decay=LearningRate/n_epochs, momentum=0.9, nesterov=True)
    
parallel_model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = ["accuracy"])

trainable_params = int(np.sum([K.count_params(p) for p in set(parallel_model.trainable_weights)]))

non_trainable_params = int(np.sum([K.count_params(p) for p in set(parallel_model.non_trainable_weights)]))

print("\nModel Stats")
print("=" * 30)
print("Total Parameters: {:,}".format((trainable_params + non_trainable_params)))
print("Non-Trainable Parameters: {:,}".format(non_trainable_params))
print("Trainable Parameters: {:,}\n".format(trainable_params))

train_folders = ["./raw/train/cancer cell/", "./raw/train/lymphocyte/", "./raw/train/plasma/"]

population_sizes = []

print("\nImages for Training")
print("=" * 30)

for folder in train_folders:
    files = glob.glob(folder + "*.png")
    n = len(files)
    print("Class: %s. " %(folder.split("/")[-2]), "Size: {:,}".format(n))
    population_sizes.append(n)

MAX = max(population_sizes)

train_images = []
train_labels = []

for index, folder in enumerate(train_folders):
     files = glob.glob(folder + "*.png")
     sample = list(np.random.choice(files, MAX))
     images = io.imread_collection(sample)
     images = [imresize(image, (139, 139)) for image in images] ### Reshape to (299, 299, 3) ###
     labels = [index] * len(images)
     train_images = train_images + images
     train_labels = train_labels + labels

train_images = np.stack(train_images)
train_images = (train_images/255).astype(np.float32) ### Standardise into the interval [0, 1] ###

train_labels = np.array(train_labels).astype(np.int32)
Y_train = to_categorical(train_labels, num_classes = np.unique(train_labels).shape[0])

valid_folders = ["./raw/validation/cancer cell/", "./raw/validation/lymphocyte/", "./raw/validation/plasma/"]


print("\nImages for Validation")
print("=" * 30)

valid_images = []
valid_labels = []

for index, folder in enumerate(valid_folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    images = [imresize(image, (139, 139)) for image in images] ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    valid_images = valid_images + images
    valid_labels = valid_labels + labels
    print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))

valid_images = np.stack(valid_images)
valid_images = (valid_images/255).astype(np.float32) ### Standardise

valid_labels = np.array(valid_labels).astype(np.int32)
Y_valid = to_categorical(valid_labels, num_classes = np.unique(valid_labels).shape[0])

print("\nBootstrapping to Balance - Training set size: %d (%d X %d)" %(train_labels.shape[0], MAX, np.unique(train_labels).shape[0]))
print("=" * 30, "\n")

#n_epochs = 70

batch_size_for_generators = 32

train_datagen = DataGenerator(rotation_range = 178, horizontal_flip = True, vertical_flip = True, shear_range = 0.6, stain_transformation = True)

train_gen = train_datagen.flow(train_images, Y_train, batch_size = batch_size_for_generators)

### VALIDATION ###

valid_datagen = DataGenerator()

valid_gen = valid_datagen.flow(valid_images, Y_valid, batch_size = batch_size_for_generators)
start = time.time()

class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_acc',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,patience=4):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, './InceptionResNetV2/'+filepath)
    file_dir = './InceptionResNetV2/log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./InceptionResNetV2/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

import time
file_path = "InceptionResNetV2_decay.hdf5"
callbacks_s = get_callbacks(file_path,top_model,patience=10)

train_steps = train_images.shape[0]//batch_size_for_generators

valid_steps = valid_images.shape[0]//batch_size_for_generators

parallel_model.fit_generator(generator = train_gen, epochs = n_epochs, steps_per_epoch = train_steps,validation_data = valid_gen, 
                        validation_steps = valid_steps, callbacks = callbacks_s,verbose=1)