import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainningData.yml")
id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        if id == 1:
            id = "Tong"
       
        cv2.putText(img,str(id),(x,y+h),font,3,(255,255,255),2,cv2.LINE_AA);

    cv2.imshow('frame',img)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()

