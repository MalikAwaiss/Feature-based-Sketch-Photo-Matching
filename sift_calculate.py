# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:38:55 2019

@author: Malik Awais
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy_sift import SIFTDescriptor 
patch_size = 16
SD = SIFTDescriptor(patchSize = patch_size)
import numpy as np
import os

def calculate():
    images = os.listdir('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn')
    #images = os.listdir('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\CUHK TRAINING\\CUHK_training_photo\\photo')
    check=1
    
    imgdist=[]
    imgdist_f=[]
#    skdist=[]
#    skdist_f=[]
#    count=0
#    dist=[]
#    match=0
#    wmatch=0
#    test=[]
    for img in images:
        im=Image.open("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\"+img)
        h,w=im.size
    
        s=16
        d=8
        n=int((w-s)/(d+1))
        m=int((h-s)/(d+1))-13
        t1=0
        ss=s
        t2=0
    #    print(n)
    #    print(m)
#        sd=[]
        c=0
        r=16
        for i in range(0,m-1):
        
    #        print(i)
            for k in range(0,n-1):
                croped=im.crop((t1,t2,s,ss))
    
                
                t1=t1+d
                s=s+d
                
    #            print(str(i)+"_outer___Inner_____"+str(k))
    
                ii=np.asarray(croped.convert('L'))
                
        #        plt.imshow(ki)
        #        plt.show()
                
        
                descriptors = SD.describe(ii)
                
        
                if descriptors.__class__.__name__== "ndarray":
        
                    imgdist.append(descriptors)
        
    
                
        
            t1=0
            t2=c+16
            
            c=t2
            
            s=16
            ss=r+16
            r=ss
        print(check)
        check=check+1
        print(img)
        temp=imgdist[0]
        for i in range(1,len(imgdist)):
            if imgdist[i].__class__.__name__== "ndarray":
        #        print("loop")
                temp=np.vstack([temp,imgdist[i]])
        for i in range(0,len(temp)):
            if temp[i][0]==-2147483648:
                for k in range(0,len(temp[i])):
                    temp[i][k]=0
        imgdist_f.append(temp)
        imgdist=[]
        
    file="discriptors.txt"
    file=open(file,"w")
    for d in imgdist_f:
        for i in d:
            for k in i:
                file.write(str(k)+"\n")
    file.close()
    file="labels.txt"
    file=open(file,"w")
    for d in images:
        file.write(d+"\n")
    file.close()
    #file="discriptors.txt"
    #count=0
    #imgenc=[]
    #img_f=[]
    #imgc=0
    #temp=[]
    #with open(file) as fp:
    #    for line in fp:
    #        
    #        dist=line.strip()
    #        temp.append(int(dist))
    #        count=count+1
    #        
    #        if count==128:
    #            imgenc.append(temp)
    #            imgc=imgc+1
    #            temp=[]
    #            count=0
    #        if imgc==300:
    #            img_f.append(imgenc)
    #            imgenc=[]
    #            imgc=0
    #for i in range(0,len(img_f)):
    #    for k in range(0,len(img_f[i])):
    #        
    #        t=img_f[i][k][0]
    #        for c in range(1,len(img_f[i][k])):
    #            t=np.vstack([t,img_f[i][k][c]])
    #        img_f[i][k]=t
    #for i in range(0,len(img_f)):
    #    t=np.transpose(img_f[i][0])
    #    for k in range(1,len(img_f[i])):
    #        t=np.vstack([t,np.transpose(img_f[i][k])])
    #    img_f[i]=t















