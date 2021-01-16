# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:12:19 2021

@author: admin
"""
import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:           
        x, y = face.left(), face.top()  
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)
        landmarks = predictor(gray, face)
        x3 = landmarks.part(18).x
        y3 = landmarks.part(18).y
        cv2.circle(frame, (x3,y3),1,(255,0,0),1)
        # print(landmarks.part(18))
        for n in range(18,28):
            x3 = landmarks.part(n).x
            y3 = landmarks.part(n).y
            cv2.circle(frame, (x3,y3),1,(255,0,0),1)
            
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) == ord('a'):
        print("pressed a")
        break

cap.release()
cv2.destroyAllWindows()