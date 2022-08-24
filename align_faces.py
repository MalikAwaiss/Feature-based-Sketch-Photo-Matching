# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import os
import cv2
import numpy as np
from skimage import color
import glob

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape_predictor_68_face_landmarks.dat", required=True,
#	help="C:\\Users\\tabis\\Downloads\\face-alignment\\face-alignment\\")
#ap.add_argument("-i", "--f-005-01", required=True,
#	help="E:\\Tabish\\Bahria University\\FYP\\DATA\\CUHK_training_photo\\photo\\New Folder\\")
#args = vars(ap.parse_args())
#path = glob.glob(r"C:\Users\ASUS\Documents\BSCS-7B\FYP\MGDB\MGDB\1hour_sketches\*.jpg")
#outpath="C:\\Users\\ASUS\\Documents\\BSCS-7B\\FYP\\MGDB\\MGDB\\1h\\*.jpg"
def align(image):

        image=cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#args["shape_predictor"]
        fa = FaceAligner(predictor, desiredFaceWidth=256)
        

        gray = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        
        # loop over the face detections
        for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
#            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(image, gray, rect)
            
#        cv2.imwrite("query.jpg",faceAligned)
#        faceAligned = cv2.imread("query.jpg",cv2.IMREAD_GRAYSCALE)
#        faceAligned=color.rgb2gray(faceAligned)
        return faceAligned
