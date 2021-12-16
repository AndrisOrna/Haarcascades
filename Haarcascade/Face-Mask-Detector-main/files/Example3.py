import cv2

#Getting Cascades
mouth_cascade=cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
face_cascade=cv2.CascadeClassifier('cascade2.xml')


#import Images

"""
Remove one commet to select an image !
"""

#img=cv2.imread('mask/have_mask_2.jpg')
#img=cv2.imread('no/no_mask_1.jpg')
img=cv2.imread('Other/other3.jpg')


gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Grayscale
face=face_cascade.detectMultiScale(gray,2,7)
mouth=mouth_cascade.detectMultiScale(gray, 2,7)

#Variables
face_msg="Face"
no_mask="Please wear a mask !"
have_mask="Thank you for wearing a mask !!!"
no_face="Face not found !"
font=cv2.FONT_HERSHEY_COMPLEX
font_scale = 0.8
font_color = (0, 0, 0)
thicknesss=1

#Face Rectangle
for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)
    cv2.rectangle(img,(x,y), (x+100,y-20), (0,255,0),-1)
    cv2.putText(img,face_msg,(x,y),font,font_scale,font_color,thicknesss,cv2.LINE_AA)

#mouth detection

#no mask
if (len(mouth)!=0 and len(face)!=0):
    cv2.rectangle(img,(0,0), (500,30), (0,0,255),-1)
    cv2.putText(img,no_mask,(0,20),font,font_scale,(0,0,0),thicknesss,cv2.LINE_AA)
#have mask
elif(len(mouth)==0 ):
    cv2.rectangle(img,(0,0), (500,30), (0,255,0),-1)
    cv2.putText(img,have_mask,(0,20),font,font_scale,(0,0,0),thicknesss,cv2.LINE_AA)
elif(len(face)==0):
    cv2.rectangle(img,(0,0), (500,30), (0,165,255),-1)
    cv2.putText(img,no_face,(0,20),font,font_scale,(0,0,0),thicknesss,cv2.LINE_AA)



cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()