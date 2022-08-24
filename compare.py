# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:30:15 2019

@author: Malik Awais
"""
import lbpclass
from sklearn.svm import LinearSVC
from PIL import Image
import argparse
import cv2
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import numpy as np
def compare(sketch):

    data=[]
    labels=[]
    distances=[]
    dt=[]
    him=[]
    hs=[]
    file="distances.txt"
    with open(file) as fp:

                for line in fp:
                    dist=line.strip()
                    data.append(float(dist))
    file="labels.txt"
    with open(file) as fp:

                for line in fp:
                    lb=line.strip()
                    labels.append(lb)
#    image = cv2.imread('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\unviewd\\'+skk,cv2.IMREAD_GRAYSCALE)
    

#    sketch=cv2.cvtColor(np.asarray(sketch), cv2.COLOR_RGB2BGR)
#    sketch=cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    
    sketch=cv2.imread(sketch,cv2.IMREAD_GRAYSCALE)
#    sketch=np.asarray(sketch)
    image=sketch
#    plt.imshow(sketch)
#    plt.show()
    hs=[]
    for b in [5,7,9,11]:
        desc = lbpclass.LocalBinaryPatterns(30, b+2)
        histt = desc.describe(image)
        hs.append(histt)
    histt=hs[0]
    for b in range(1,len(hs)):
        histt=np.vstack((histt,hs[b]))
    #print(sktname)
    images=os.listdir('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn')
    for img in images:
        gray = cv2.imread('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\'+img,cv2.IMREAD_GRAYSCALE)
        
    #        hist = desc.describe(gray)
        him=[]
        for b in [5,7,9,11]:
            desc = lbpclass.LocalBinaryPatterns(30, b)
            hist = desc.describe(gray)
            him.append(hist)
        hist=him[0]
        for b in range(1,len(him)):
            hist=np.vstack((hist,him[b]))
        hist=np.vstack((hist,histt))
        d=LA.norm(hist)
        for a in data:
            distances.append(abs(d-a))
        dt.append(distances)
        distances=[]
    ind=[]
    ext=[]
    for dv in dt:
        
        value=min(dv)
        index=dv.index(value)
        ext.append(value)
        ind.append(index)
    value=min(ext)
    index=ext.index(value)
    index=ind[index]
    result=Image.open('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\'+labels[index])
    return result
#imn = cv2.imread('f2-007-01-sz1.jpg',cv2.IMREAD_GRAYSCALE)
#imn=Image.open('f2-007-01-sz1.jpg').convert('L')
#image=compare(imn)
#plt.imshow(image)
#plt.show()