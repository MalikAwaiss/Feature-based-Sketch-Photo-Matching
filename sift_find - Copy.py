# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:22:39 2019

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
def find(im,img_f):
    sktdist=[]
    im=Image.open(im)
    distt=[]
    h,w=im.size
    labels=[]
    s=16
    d=8
    n=int((w-s)/(d+1))
    m=int((h-s)/(d+1))-13
    t1=0
    ss=s
    t2=0
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
    
                sktdist.append(descriptors)
    

            
    
        t1=0
        t2=c+16
        
        c=t2
        
        s=16
        ss=r+16
        r=ss
    
#    print(img)
    temp=sktdist[0]
    for i in range(1,len(sktdist)):
        if sktdist[i].__class__.__name__== "ndarray":
    #        print("loop")
            temp=np.vstack([temp,sktdist[i]])
    for i in range(0,len(temp)):
        if temp[i][0]==-2147483648:
            for k in range(0,len(temp[i])):
                temp[i][k]=0
    sktdist=temp
#    file="discriptors.txt"
#    count=0
#    imgenc=[]
#    img_f=[]
#    imgc=0
#    temp=[]
#    with open(file) as fp:
#        for line in fp:
#            
#            dist=line.strip()
#            temp.append(int(dist))
#            count=count+1
#            
#            if count==128:
#                imgenc.append(temp)
#                imgc=imgc+1
#                temp=[]
#                count=0
#            if imgc==300:
#                img_f.append(imgenc)
#                imgenc=[]
#                imgc=0
#    for i in range(0,len(img_f)):
#        for k in range(0,len(img_f[i])):
#            
#            t=img_f[i][k][0]
#            for c in range(1,len(img_f[i][k])):
#                t=np.vstack([t,img_f[i][k][c]])
#            img_f[i][k]=t
#    for i in range(0,len(img_f)):
#        t=np.transpose(img_f[i][0])
#        for k in range(1,len(img_f[i])):
#            t=np.vstack([t,np.transpose(img_f[i][k])])
#        img_f[i]=t
    file="labels.txt"
    with open(file) as fp:
        for line in fp:
            labels.append(line.strip())
    for d in img_f:
        distt.append(np.linalg.norm(d-sktdist))
    min_d=min(distt)
    ind=distt.index(min_d)
    result=Image.open("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\"+labels[ind])
    return result
def find3(im,img_f):
    sktdist=[]
    im=Image.open(im)
    distt=[]
    h,w=im.size
    labels=[]
    s=16
    d=8
    n=int((w-s)/(d+1))
    m=int((h-s)/(d+1))-13
    t1=0
    ss=s
    t2=0
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
    
                sktdist.append(descriptors)
    

            
    
        t1=0
        t2=c+16
        
        c=t2
        
        s=16
        ss=r+16
        r=ss
    
#    print(img)
    temp=sktdist[0]
    for i in range(1,len(sktdist)):
        if sktdist[i].__class__.__name__== "ndarray":
    #        print("loop")
            temp=np.vstack([temp,sktdist[i]])
    for i in range(0,len(temp)):
        if temp[i][0]==-2147483648:
            for k in range(0,len(temp[i])):
                temp[i][k]=0
    sktdist=temp
#    file="discriptors.txt"
#    count=0
#    imgenc=[]
#    img_f=[]
#    imgc=0
#    temp=[]
#    with open(file) as fp:
#        for line in fp:
#            
#            dist=line.strip()
#            temp.append(int(dist))
#            count=count+1
#            
#            if count==128:
#                imgenc.append(temp)
#                imgc=imgc+1
#                temp=[]
#                count=0
#            if imgc==300:
#                img_f.append(imgenc)
#                imgenc=[]
#                imgc=0
#    for i in range(0,len(img_f)):
#        for k in range(0,len(img_f[i])):
#            
#            t=img_f[i][k][0]
#            for c in range(1,len(img_f[i][k])):
#                t=np.vstack([t,img_f[i][k][c]])
#            img_f[i][k]=t
#    for i in range(0,len(img_f)):
#        t=np.transpose(img_f[i][0])
#        for k in range(1,len(img_f[i])):
#            t=np.vstack([t,np.transpose(img_f[i][k])])
#        img_f[i]=t
    file="labels.txt"
    with open(file) as fp:
        for line in fp:
            labels.append(line.strip())
    for d in img_f:
        distt.append(np.linalg.norm(d-sktdist))
    ind=[]
    for k in range(0,3):
        min_d=min(distt)
        index=distt.index(min_d)
        distt[index]=distt[index]+distt[index]
        ind.append(index)
    result=[]
    for k in ind:
#        result.append(Image.open("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\"+labels[k]))
#        print(index)
        t=cv2.imread("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\"+labels[k])
        result.append(t)
#    for k in range(0,len(result)):
#        result[k] = cv2.cvtColor(numpy.array(result[k]), cv2.COLOR_RGB2BGR)
    results=cv2.hconcat([result[0],result[1],result[2]])
    cv2.imwrite("result.jpg",results)
    results=Image.open("result.jpg")
#    results=Image.fromarray(results)
    return results

def find10(im,img_f):
    sktdist=[]
    im=Image.open(im)
    distt=[]
    h,w=im.size
    labels=[]
    s=16
    d=8
    n=int((w-s)/(d+1))
    m=int((h-s)/(d+1))-13
    t1=0
    ss=s
    t2=0
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
    
                sktdist.append(descriptors)
    

            
    
        t1=0
        t2=c+16
        
        c=t2
        
        s=16
        ss=r+16
        r=ss
    
#    print(img)
    temp=sktdist[0]
    for i in range(1,len(sktdist)):
        if sktdist[i].__class__.__name__== "ndarray":
    #        print("loop")
            temp=np.vstack([temp,sktdist[i]])
    for i in range(0,len(temp)):
        if temp[i][0]==-2147483648:
            for k in range(0,len(temp[i])):
                temp[i][k]=0
    sktdist=temp
#    file="discriptors.txt"
#    count=0
#    imgenc=[]
#    img_f=[]
#    imgc=0
#    temp=[]
#    with open(file) as fp:
#        for line in fp:
#            
#            dist=line.strip()
#            temp.append(int(dist))
#            count=count+1
#            
#            if count==128:
#                imgenc.append(temp)
#                imgc=imgc+1
#                temp=[]
#                count=0
#            if imgc==300:
#                img_f.append(imgenc)
#                imgenc=[]
#                imgc=0
#    for i in range(0,len(img_f)):
#        for k in range(0,len(img_f[i])):
#            
#            t=img_f[i][k][0]
#            for c in range(1,len(img_f[i][k])):
#                t=np.vstack([t,img_f[i][k][c]])
#            img_f[i][k]=t
#    for i in range(0,len(img_f)):
#        t=np.transpose(img_f[i][0])
#        for k in range(1,len(img_f[i])):
#            t=np.vstack([t,np.transpose(img_f[i][k])])
#        img_f[i]=t
    file="labels.txt"
    with open(file) as fp:
        for line in fp:
            labels.append(line.strip())
    for d in img_f:
        distt.append(np.linalg.norm(d-sktdist))
    ind=[]
    for k in range(0,10):
        min_d=min(distt)
        index=distt.index(min_d)
        distt[index]=distt[index]+distt[index]
        ind.append(index)
    result=[]
    for k in ind:
#        result.append(Image.open("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\"+labels[k]))
#        print(index)
        t=cv2.imread("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\nn\\"+labels[k])
        result.append(t)
#    for k in range(0,len(result)):
#        result[k] = cv2.cvtColor(numpy.array(result[k]), cv2.COLOR_RGB2BGR)
    results=cv2.hconcat([result[0],result[1],result[2]])
    resultss=cv2.hconcat([result[3],result[4],result[5]])
    resultsss=cv2.hconcat([result[6],result[7],result[8]])
    results=cv2.vconcat([results,resultss])
    results=cv2.vconcat([results,resultsss])
    cv2.imwrite("result.jpg",results)
    results=Image.open("result.jpg")
    return results

            
            
            
            