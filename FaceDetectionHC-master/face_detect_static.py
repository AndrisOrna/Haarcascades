#import numpy as np
import cv2
import sys

# Get user supplied values
imagePath = sys.argv[0]

# Load the Haar Cascade 
cascPath = 'haarcascade_frontalface_default.xml'
# left eye Cascade
cascPath1 = 'haarcascade_lefteye_2splits.xml'
# right eye Cascade
cascPath2 = 'haarcascade_righteye_2splits.xml'



# Create the Haar Cascade
faceCascade = cv2.CascadeClassifier(cascPath)
leftEyeCascade = cv2.CascadeClassifier(cascPath1)
rightEyeCascade = cv2.CascadeClassifier(cascPath2)

# Read the Image
img = cv2.imread('bfZUt.jpg')

# Convert to Gray-Scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#red = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the Faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


roi_gray = gray[y:y+h, x:x+w]
roi_color = img[y:y+h, x:x+w]
lefteyes = leftEyeCascade.detectMultiScale(roi_gray)
righteyes = rightEyeCascade.detectMultiScale(roi_gray)


for (ex,ey,ew,eh) in lefteyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
    
# for (xe,ye,we,he) in righteyes:
#     cv2.rectangle(roi_color,(xe,ye),(xe+we,ye+he),(0,124,0),2)

cv2.imshow('img',img)
k = cv2.waitKey(0) 
    
cv2.imshow('Left eye found', img)
cv2.waitKey(0)

