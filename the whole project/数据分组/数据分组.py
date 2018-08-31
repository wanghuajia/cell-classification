# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc
import random

def creat_Img(imgDir1):
    dir_1 = os.listdir('./' + imgDir1)
    print(dir_1)
    for i in range(len(dir_1)):
        n=0
        m=0
        img_list1 = os.listdir('./'+imgDir1+'/'+dir_1[i])
        print(dir_1[i]+' : '+ str(len(img_list1)))
        for j in range(len(img_list1)):
            img1 = cv2.imread('./' + imgDir1+'/'+dir_1[i] + '/' + img_list1[j])
            img_save_dir = './raw/train/'+dir_1[i]
            a = random.randint(1,4)
            if (a%2==0)and(n<300):
                img_save_dir = './raw/validation/'+dir_1[i]
                n = n+1
            if (a%3==0)and(m<300):
                img_save_dir = './raw/test/'+dir_1[i]
                m = m+1
            if not os.path.exists(img_save_dir):os.makedirs(img_save_dir)
            #print(os.path.splitext(img_list1[j])[0])
            cv2.imwrite(img_save_dir+'/'+os.path.splitext(img_list1[j])[0]+'.png',img1)
            
if __name__ == '__main__':
    creat_Img('breast_70')
