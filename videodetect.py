import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture('carvideo.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)

#incase video unable to open
if (cap.isOpened()==False):
    print('Error Reading video')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#rio extract specific part from image/video
        roigrey=frame[y:y+h,x:x+w]
        blurr=cv2.blur(roigrey,(20,20))
        frame[y: y+h, x:x+w] = blurr

    if ret == True:
        cv2.imshow('Video',frame)
    #waitkey hold the video to play and pressing 'q' will exit the video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()
