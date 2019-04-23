# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:43:38 2019

@author: I'm Harshil
"""


import numpy as np
import cv2
cap=cv2.VideoCapture('Pedestrian.mp4')
kernel=np.ones((20,20),np.uint8)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
i=0
while(1):
    ref, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    if ref==True:
        contours,hierachy=cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if(area>2500):
                i=i+1
                
                x,y,w,h = cv2.boundingRect(contour)
                frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        resize1=cv2.resize(fgmask,(400,400))    
        cv2.imshow('frame',resize1)
        resize2=cv2.resize(frame,(400,400))
        cv2.imshow('original',resize2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()