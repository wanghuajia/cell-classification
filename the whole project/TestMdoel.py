# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:05:19 2018

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
from keras.applications.xception import Xception
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

base=load_model('Xception_model(71).h5')

top_model = Sequential()
top_model.add(base)
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax',kernel_initializer=Orthogonal()))
top_model.summary()

top_model.compile(optimizer = "SGD", loss = "categorical_crossentropy", metrics = ["accuracy"])

top_model.load_weights('./Xception/Xception_decay.hdf5')

import cv2
import numpy as np
from sklearn import cross_validation, metrics

test_folders = ['./raw/test/cancer cell/', './raw/test/lymphocyte/', './raw/test/plasma cell/']

print("\nImages for Testing")
print("=" * 30)

test_images = []
test_labels = []

for index, folder in enumerate(test_folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    images = [imresize(image, (71, 71)) for image in images] ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    test_images = test_images + images
    test_labels = test_labels + labels
    print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))

print("\n")

test_images = np.stack(test_images)
test_images = (test_images/255).astype(np.float32) ### Standardise

test_labels = np.array(test_labels).astype(np.int32)
Y_test = to_categorical(test_labels, num_classes = np.unique(test_labels).shape[0])

### 

test_loss, test_accuracy = top_model.evaluate(test_images, Y_test, batch_size = 50)

print("\nTest Loss: %.3f" %(test_loss))
print("Test Accuracy: %.3f" %(test_accuracy))
print("=" * 30, "\n")

print("Classification Report")
print("=" * 30, "\n")

posteriors = top_model.predict(test_images, batch_size = 32)
predictions = np.argmax(posteriors, axis = 1)

cr = classification_report(test_labels, predictions, target_names = ["cancer_cell","lymphocyte","plasma_cell"], digits = 3)
print(cr, "\n")

print("Confusion Matrix")
print("=" * 30, "\n")

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predictions)
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size=15)
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["cancer_cell","lymphocyte","plasma_cell"],rotation=45, size=10)
plt.yticks(tick_marks, ["cancer_cell","lymphocyte","plasma_cell"],size=10)
plt.tight_layout()
plt.ylabel('Actual label',size=15)
plt.xlabel('Predicted label',size=15)
width, height=cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
plt.show()
#print(cm, "\n")