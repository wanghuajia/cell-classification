# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:36:22 2018

@author: Biototem_1
"""
import scipy.io as scio  
from skimage import io
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import glob
from keras.models import *
import time 

Xception_model=load_model('Xception.h5')
InceptionResNetV2_model=load_model('InceptionResNetV2.h5')
InceptionV3_model = load_model('InceptionV3.h5')

def cell_cut(folders_outline,folders_annatated,Xception_model,InceptionResNetV2_model,InceptionV3_model,own_slide):
    img_list_outline= os.listdir('./'+folders_outline)
    img_list_annatated= os.listdir('./'+folders_annatated)
    miss=[]
    #time_sum= []
    for m in range(1281,1291):
        #start_time = time.time()
        if img_list_outline[m]=='24_32769_43521(1).npy':
            continue
        data = np.load('./'+folders_outline+'/'+img_list_outline[m])
        data = np.array(data, np.uint8)
        img = cv2.imread('./'+folders_annatated + '/' + os.path.splitext(img_list_outline[m])[0]+'.png')
        if img is None:
            print( os.path.splitext(img_list_outline[m])[0])
            miss.append(img_list_outline[m])
            continue

        for n in range(data.shape[2]):
            ret,thresh = cv2.threshold(data[:,:,n],0,255,0)
            image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                #x,y,w,h = cv2.boundingRect(contours[i])
                #slide = max(w,h)
                M = cv2.moments(contours[i])
                try:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                except ZeroDivisionError:
                    continue
                else:
                    x1 = cx-math.ceil(own_slide/2)
                    y1 = cy-math.ceil(own_slide/2)
                    x2 = cx+math.ceil(own_slide/2)
                    y2 = cy+math.ceil(own_slide/2)
                    if (cx<math.ceil(own_slide/2))or(cy<math.ceil(own_slide/2))or(cx>(513-math.ceil(own_slide/2)))or(cy>(513-math.ceil(own_slide/2))):
                        continue
                    cropImg = img[y1:y2, x1:x2]
                    I = cv2.cvtColor(cropImg,cv2.COLOR_BGR2RGB)
                    I=cv2.resize(I,(139,139))
                    I = I/255
                    I = np.expand_dims(I,axis=0)
                    Xception_preds=Xception_model.predict(I)
                    InceptionV3_preds=InceptionV3_model.predict(I)
                    InceptionResNetV2_preds=InceptionResNetV2_model.predict(I)
                    semple = Xception_preds + InceptionV3_preds + InceptionResNetV2_preds
                    preds = np.argmax(semple)
                    preds_pro = max(max(semple))
                    preds_pro = round(preds_pro/3,2)
                    if preds==0:
                        cv2.drawContours(img,contours,i,(0,255,0),1)
                    if preds==1:
                        cv2.drawContours(img,contours,i,(0,0,255),1)
                    if preds==2:
                        cv2.drawContours(img,contours,i,(0,255,255),1)
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,str(preds_pro),(cx,cy), font, 0.4,(255,255,255),1)
                    #k = k+1
        dir_name='./predict_map'
        if not os.path.exists(dir_name):os.makedirs(dir_name)
        cv2.imwrite(dir_name+'/'+os.path.splitext(img_list_outline[m])[0]+'.png',img)
        #cost_time = time.time()-start_time
        #time_sum.append(cost_time)
    #return time_sum
if __name__ == '__main__':
    time_04=cell_cut("new_image_classify_v4_67_mask","1290 New Patches",Xception_model,InceptionResNetV2_model,InceptionV3_model,70)
    
#import pandas as pd
#time_04
#a=np.hstack((time_01,time_02,time_04))
#a=pd.DataFrame(a)
#a.to_csv ("time.csv" , encoding = "utf-8")
