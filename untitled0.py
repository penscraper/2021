# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:12:19 2021

@author: admin
"""
import cv2
import numpy as np
import dlib
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()


# keyboard.press(Key.down)
# keyboard.release(Key.down)
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((15, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(18, 28):
		coords[i-18] = (shape.part(i).x, shape.part(i).y)
   
        # cv2.circle(frame, (coords[i][0],coords[i][1]),1,(255,0,0),2)
	# return the list of (x, y)-coordinates
	return coords


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

coords = np.zeros((10, 2), dtype="int")
avgL = np.zeros(2, dtype="int")
avgR = np.zeros(2, dtype="int")
# nosept = np.zeros(2, dtype="int")
# print(coords)
# if True:
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:          
        # x, y = face.left(), face.top()  
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)
        landmarks = predictor(gray, face)
        # coordss = shape_to_np(landmarks)
        # print(coordss)
        # x3 = landmarks.part(18).x
        # y3 = landmarks.part(18).y
        # cv2.circle(frame, (x3,y3),1,(255,0,0),1)
        # print(landmarks.part(18))
        for n in range(18,28):
           coords[n-18] = (landmarks.part(n).x,landmarks.part(n).y)
           
        np.mean(coords[0:5], axis = 0, out=avgL)
        np.mean(coords[5:10], axis = 0, out=avgR)
        # print(avgL)
        # print(avgR)
        nosept = (landmarks.part(28).x,landmarks.part(28).y)
        noseptx, nosepty = (landmarks.part(28).x,landmarks.part(28).y)
        
        avgdistance = np.linalg.norm(avgL-nosept)
        # print(avgdistance)
        # cv2.circle(frame, (avgL[0],avgL[1]),1,(255,0,0),2)
        cv2.line(frame,(avgL[0],avgL[1]),(noseptx,nosepty),(0,255,0),2)
        
        if (avgdistance <= 38):
            keyboard.press(Key.down)
            print('down')
        elif(avgdistance >38 or avgdistance < 45):
            keyboard.release(Key.down)
            keyboard.release(Key.up)
            print('stop')
        elif (avgdistance >= 45):
            keyboard.press(Key.up)
            print('up')
        # cv2.circle(frame, (avgR[0],avgR[1]),1,(255,0,0),2)
        # cv2.circle(frame, (landmarks.part(28).x,landmarks.part(28).y),1,(255,0,0),2)
        
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) == ord('a'):
        print("pressed a")
        break

cap.release()
cv2.destroyAllWindows()