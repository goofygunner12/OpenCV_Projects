#Import required packages
import numpy as np
import cv2
import math

#load the classifier files: full body, upperbody, lowerbody, face and eyes.
fullBody_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_fullbody.xml')
upperBody_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_upperbody.xml')
lowerBody_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_lowerbody.xml')
face_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_eye.xml')

#Read the Video file
img = cv2.VideoCapture('C:\Users\Vishwas\Downloads\Chrome Downloads\Moment Nepal earthquake hit - BBC News.mp4')
#img = cv2.VideoCapture('C:\Users\Vishwas\Desktop\AINT510 Videos\Mov_20150426144643.avi')
#img = cv2.VideoCapture('C:\Users\Vishwas\Desktop\AINT510 Videos\Mov_20150426145215.avi')

#Used for Video capture in real-time
#nFrames = int(  cv2.cv.GetCaptureProperty( img, cv2.cv.CV_CAP_PROP_FRAME_COUNT ) )
#fps = cv2.cv.GetCaptureProperty( img, cv2.cv.CV_CAP_PROP_FPS )
#waitPerFrameInMillisec = int( 1/fps * 1000/1 )
#print 'Num. Frames = ', nFrames
#print 'Frame Rate = ', fps, ' frames per sec'

#converter for video writer
fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
video = cv2.VideoWriter('vish.avi',fourcc, 10, (640,480),1)
#video = cv2.VideoWriter("C:\Users\Vishwas\Videos\vish.avi", cv2.cv.CV_FOURCC('F','M','P', '4'), 15, (1536, 1536), 1)

#Main loop where the classifier functions
while True:
#for f in xrange( nFrames ):

    #frameImg = cv2.cv.QueryFrame( img )
    ret, frame = img.read()
    gray =  cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #COLOR_BGR2GRAY)
    #Detect full body
    full_body = fullBody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(25, 25),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in full_body:
        print 'Body detected'
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.putText(frame,"Body detected",(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5 ,(0,0,0))
    #Detect lower body
    lower_body = lowerBody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(25, 25),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in lower_body:
        print 'Legs detected'
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,"Legs detected",(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5 ,(0,255,0))
    #Detect Upper body
    upper_body = upperBody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(25, 25),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in upper_body:
        print 'Upper body detected'
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(frame,"Upper body detected",(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5 ,(0,255,255))
    #Detect face and eyes    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cv2.putText(frame,"Face detected",(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5 ,(0,255,255))
        for (ex,ey,ew,eh) in eyes:            
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0),2)
            cv2.circle(roi_color,((ex+ex+ew)/2,(ey+eh+ey)/2), 3, (0,0,255), -1)
    #Write the video into a file and Display the frame
    #out.write(frame)
    video.write(frame)
    cv2.imshow('Video',frame)
    #cv2.waitKey(25)

    #Condition to check for a key to quit from the loop
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
#Closing functions
img.release()
video.release()
print 'Exiting the Video'
cv2.destroyAllWindows()
