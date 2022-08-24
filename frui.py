# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:09:19 2019

@author: Malik Awais
"""

import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import face_recognition
import cv2
import numpy as np

def matchsketch(sketch):
    
    images=os.listdir('C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\CUHK TRAINING\\CUHK_training_photo\\a\\a')
#    sketch = face_recognition.load_image_file(sketchpath)
    sketchen=face_recognition.face_encodings(sketch)
    results=[]
    for image in images:
            # load the image
        #    current_image = face_recognition.load_image_file("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\CUHK TRAINING\\CUHK_training_photo\\photo\\a\\a\\" + image)
    #        current_image = face_recognition.load_image_file("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\CUHK TRAINING\\CUHK_training_photo\\im\\" + image)
            current_image = face_recognition.load_image_file("C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\CUHK TRAINING\\CUHK_training_photo\\a\\a\\" + image)
#            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            # encode the loaded image into a feature vector
            current_image_encoded = face_recognition.face_encodings(current_image,num_jitters=5)[0]
            # match your image with the image and check if it matches
            result = face_recognition.compare_faces(
                [sketchen], current_image_encoded,tolerance=0.65
                )
            # check if it was a match
            tem=result[0]
            if tem:
                ii="C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\CUHK TRAINING\\CUHK_training_photo\\a\\a\\"+image
                results.append(ii)
#                print ("Matched: " + image)
    return results