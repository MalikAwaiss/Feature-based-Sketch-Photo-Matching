# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:10:45 2019

@author: Malik Awais
"""


import lbpclass
from sklearn.svm import LinearSVC

import argparse
import sys
import cv2
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import numpy as np
#desc = lbpclass.LocalBinaryPatterns(40, 20)
def compute(sketche,imag):
    data=[]
    labels=[]
    him=[]
    hs=[]
    sketches = os.listdir(sketche)
    images = os.listdir(imag)
    ind=0
    for image in images:
#        print(image)
        gray = cv2.imread(imag+'\\'+image,cv2.IMREAD_GRAYSCALE)
        sketch = cv2.imread(sketche+'\\'+sketches[ind],cv2.IMREAD_GRAYSCALE)
    #    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        him=[]
        hs=[]
        for b in [5,7,9,11]:
            desc = lbpclass.LocalBinaryPatterns(30, b)
            hist = desc.describe(gray)
            desc = lbpclass.LocalBinaryPatterns(30, b+2)
            histt = desc.describe(sketch)
            him.append(hist)
            hs.append(histt)
        hist=him[0]
        histt=hs[0]
        for b in range(1,len(him)):
            hist=np.vstack((hist,him[b]))
            histt=np.vstack((histt,hs[b]))
        hist=np.vstack((hist,histt))
        data.append(LA.norm(hist))
        labels.append(image)
        ind=ind+1
    file="distances.txt"
    file=open(file,"w")
    for d in data:
        file.write(str(d) + "\n")
    file.close()
    file="labels.txt"
    file=open(file,"w")
    for d in labels:
        file.write(str(d) + "\n")
    file.close()
#compute('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nnn','C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn')