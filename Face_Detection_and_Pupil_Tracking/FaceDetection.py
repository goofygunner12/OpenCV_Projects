import numpy as np
import cv2
import math

eye_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
mouth_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml')
img = cv2.VideoCapture(0)

while True:
    ret, frame = img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        for (ex,ey,ew,eh) in eyes:
            
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0),2)
            cv2.circle(roi_color,((ex+ex+ew)/2,(ey+eh+ey)/2), 3, (0,0,255), -1)
                    

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
img.release()
cv2.destroyAllWindows()
