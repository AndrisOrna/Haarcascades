# -*- coding: utf-8 -*-


# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", type=str,default="C:/Users/A00129244/Documents/GitHub/Haarcascades/Haarcascade/video/video2.mp4",
    help="path to input image")
ap.add_argument("-c", "--cascades", type=str, default="C:/Users/A00129244/Documents/GitHub/Haarcascades/Haarcascade/",
    help="path to input directory containing haar cascades")
args = vars(ap.parse_args())

# initialize a dictionary that maps the name of the haar cascades to
# their filenames
detectorPaths = {
    "face": "cars.xml",
    "eyes": "two_wheeler.xml",
    "smile": "pedestrian.xml",
}
# initialize a dictionary to store our haar cascade detectors
print("[INFO] loading haar cascades...")
detectors = {}
# loop over our detector paths
for (name, path) in detectorPaths.items():
    # load the haar cascade from disk and store it in the detectors
    # dictionary
    path = os.path.sep.join([args["cascades"], path])
    detectors[name] = cv2.CascadeClassifier(path)
    
    # initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap=cv2.VideoCapture(args["video"])
# loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it, and convert it
    # to grayscale
    ret,frame = cap.read()
    if (type(frame) == type(None)):
        break
    # change width to slow down video and resize
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform face detection using the appropriate haar cascade
    faceRects = detectors["face"].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    eyeRects = detectors["eyes"].detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=10,
        minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
    # apply smile detection to the face ROI
    smileRects = detectors["smile"].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=10,
        minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the face bounding boxes
    for (fX, fY, fW, fH) in faceRects:
        # extract the face ROI
        faceROI = gray[fY:fY+ fH, fX:fX + fW]
        # apply eyes detection to the face ROI
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
            (0, 255, 0), 2)
        # loop over the eye bounding boxes
    for (eX, eY, eW, eH) in eyeRects:
        # draw the eye bounding box
        ptA = (eX, eY)
        ptB = ( eX + eW, eY + eH)
        cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
    # loop over the smile bounding boxes
    for (sX, sY, sW, sH) in smileRects:
        # draw the smile bounding box
        ptA = ( sX,  sY)
        ptB = ( sX + sW,  sY + sH)
        cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)
    # draw the face bounding box on the frame
        
        
        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()