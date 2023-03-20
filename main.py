import cv2
import os
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

imgbackground = cv2.imread("Resources/Background.png")
#import modes images kedalam list
folder_lajur_mode = "Resources/Modes"
modePath = os.listdir(folder_lajur_mode)
imgModeList = []

for path in modePath:
    imgModeList.append(cv2.imread(os.path.join(folder_lajur_mode, path)))
# print(modePath)
# print(len(imgModeList))

while True:
    success, img = cam.read()

    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

    imgbackground[162:162+480, 55:55+640] = img
    imgbackground[44:44+633, 808:808+414] = imgModeList[1]

    # cv2.imshow("Webcam",img)
    # cam.release()
    # cv2.destroyAllWindows()
    cv2.imshow("Background", imgbackground)
    cv2.waitKey(1)
