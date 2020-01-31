import cv2,os
import numpy as np
from PIL import Image 
import serial
#ser = serial.Serial('COM7', 9600)
#for ip camera
#for ip camera
url='http://192.168.43.1:8080/video'
#cam=cv2.VideoCapture(url)

#for webcam
buzzercount=0
doorlock=0


#cam=cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/traininData.yml")
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL


while (True):
    
    ret, imc =cam.read()
    im= cv2.flip(imc, 1)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = rec.predict(gray[y:y+h,x:x+w])
        buzzercount= buzzercount+ 1
        if buzzercount==2:
            print(" ")
            #ser.write('h'.encode())

        #print(buzzercount,"\n")
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(0,0,255),2)
        print(conf,"\n")
        if(conf<30):
            if(doorlock==5):
                print(" ")
               # ser.write('g'.encode())

            
          
            nbr_predicted='kushal'
            doorlock=doorlock + 1
            buzzercount=0
            cv2.putText(im,str(nbr_predicted), (x,y+h),font, 1.1, (0,0,255))
        
    cv2.imshow('Face recog',im)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()



