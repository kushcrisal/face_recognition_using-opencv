import cv2
import os
#for ip camera
url='http://192.168.43.1:8080/video'
#cam=cv2.VideoCapture(0)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# for webcam
#cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

i=4837

offset=50
name=input('enter your id')

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(100)
    if i>5000:
        cam.release()
        cv2.destroyAllWindows()
        break