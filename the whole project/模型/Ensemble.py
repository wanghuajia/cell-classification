# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:03:49 2018

@author: Biototem_1
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.initializers import Orthogonal
from keras.models import Model
import argparse
import keras
import os
from pathlib import Path
import sys
import glob
import numpy as np
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
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten

Xception_model=load_model('Xception_complete.h5')
InceptionV3_model=load_model('InceptionV3_complete.h5')
InceptionResNetV2_model =load_model('InceptionResNetV2_complete.h5')

def model_predict(Xception_model, InceptionV3_model, InceptionResNetV2_model, imgDir):
    n_samples = sum([len(files) for root,dirs,files in os.walk(imgDir)])
    np.set_printoptions(suppress=True)
    dir_1 = sorted(os.listdir('./' + imgDir))
    array_pred = np.empty((n_samples,1))
    array_y= np.empty((n_samples,1))
    n=0
    equal=0
    for i in range(len(dir_1)):
        img_list = os.listdir('./'+imgDir+'/'+dir_1[i])
        print(dir_1[i]+' : '+ str(len(img_list)))
        sum_01=0
        for j in range(len(img_list)):
            I = io.imread('./' + imgDir+'/'+dir_1[i] + '/' + img_list[j])
            #I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
            I=cv2.resize(I,(139,139))
            I = I/255
            I = np.expand_dims(I,axis=0)
            Xception_preds=Xception_model.predict(I)
            Xception_pro=np.max(Xception_preds)

            InceptionV3_preds=InceptionV3_model.predict(I)
            InceptionV3_pro=np.max(InceptionV3_preds)

            InceptionResNetV2_preds=InceptionResNetV2_model.predict(I)
            InceptionResNetV2_pro=np.max(InceptionResNetV2_preds)
            #if i!=2:
            if (Xception_pro>InceptionV3_pro)and(Xception_pro>InceptionResNetV2_pro):
                pred_1=np.argmax(Xception_preds)

            elif (InceptionV3_pro>Xception_pro)and(InceptionV3_pro>InceptionResNetV2_pro):
                pred_1=np.argmax(InceptionV3_preds)
            elif (InceptionResNetV2_pro>Xception_pro)and(InceptionResNetV2_pro>InceptionV3_pro):
                pred_1=np.argmax(InceptionResNetV2_preds)
            else:
                equal = equal+1
                pred_1=np.argmax(Xception_preds)
            #elif i==2:
            #pred_1=np.argmax(Xception_preds)
            array_pred[n] = pred_1
            array_y[n] = i
            n=n+1
            if i==pred_1:
                sum_01=sum_01+1
        print(sum_01)
    print(equal)
    return array_pred,array_y

if __name__ == '__main__':
    y_pred ,y_test = model_predict(Xception_model,InceptionV3_model,InceptionResNetV2_model,'raw/test')
    
from sklearn.metrics import classification_report, confusion_matrix    
print("Classification Report")
print("=" * 30, "\n")
cr = classification_report(y_test, y_pred, target_names =  ["cancer cell","lymphocyte","plasma cell"], digits = 3)
print(cr, "\n")

print("Confusion Matrix")
print("=" * 30, "\n")

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size=15)
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks,  ["cancer cell","lymphocyte","plasma cell"],rotation=45, size=10)
plt.yticks(tick_marks,  ["cancer cell","lymphocyte","plasma cell"],size=10)
plt.tight_layout()
plt.ylabel('Actual label',size=15)
plt.xlabel('Predicted label',size=15)
width, height=cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
plt.show()