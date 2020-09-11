import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

img = cv2.imread('newcar.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roigrey=img[y:y+h,x:x+w]
    blurr=cv2.blur(roigrey,(25,25))
    img[y: y+h, x:x+w] = blurr

cv2.imshow('faces',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
