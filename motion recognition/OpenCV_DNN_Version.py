#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import mediapipe as mp
import time
import glob


############

import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import tensorflow as tf
import PIL
from PIL import Image 

model_name='res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name='deploy.prototxt.txt'
model77=tf.keras.models.load_model('last_model.hdf5')
categories=["netural","not_understand","understand"]


# In[ ]:


def detection_preprocessing(image, h_max=360):
    h, w, _ = image.shape
    if h > h_max:
        ratio = h_max / h
        w_ = int(w * ratio)
        image = cv2.resize(image, (w_,h_max))
    return image

def resize_face(face):
    x = tf.convert_to_tensor(face)
    return tf.image.resize(x, (299,299))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x


##color
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)
points_list =[(200, 300), (150, 150), (400, 200)]

# ----------------------------------------------------------------------------

def detectAndDisplay(frame):
    frame_emo = frame
    detector=cv2.dnn.readNetFromCaffe(prototxt_name,model_name)
    original_size = frame_emo.shape
    target_size = (300,300)
    
    
    image = cv2.resize(frame_emo, target_size)
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
    imageBlob = cv2.dnn.blobFromImage(image = image)
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
    #display( detections_df)
    
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence'] >= 0.70]
    
    for i, instance in detections_df.iterrows():

        confidence_score = str(round(100*instance["confidence"], 2))+" %"

        left = int(instance["left"] * 300)
        bottom = int(instance["bottom"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)

        detected_face = frame_emo[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

        #감정값 예측하기

        emo_data=Image.fromarray(detected_face[:,:,::-1])
        emo=np.asarray(emo_data)
        emo=cv2.resize(emo, (299,299))
        emo=emo.reshape((-1, 299,299,3)
        test = model77.predict(emo)
        y=model77.predict(emo).argmax()



#---------------------------------------------------------------------------------------

hand_result = None
frame_counter =0
CEF_COUNTER=0
COUNTER =0
dCOUNTER=0
closed_time=0
closed_time1=0
sleep = 0
v_start=0
c_end=0
vanish=0
c_start=0
GONE=0
CLOSED_EYES_FRAME =60
FONTS =cv2.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv2.VideoCapture(0)

#video
# video_file = r"frame 2022-03-24 18-42-46.mp4"  # 비디오 경로
# camera = cv2.VideoCapture(video_file)



#####hand detection settings

csv_file = 'gesture_train_test9.csv'
max_num_hands = 10

study_gesture = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 9:'two', 10:'ok', 11:'x', 12:'three'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt(csv_file, delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)


# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = max(rhDistance,lhDistance)/rvDistance
    leRatio = max(rhDistance,lhDistance)/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time=time.time()
    # starting Video loop here.
    while True:

        frame_counter +=1 # frame counter
        
        ret, frame = camera.read() # getting frame from camera 
        
        if not ret: 
            break # no more frames break
        #  resizing frame
        
    # -------------------------------------------------------------------------------------    
         # face detection
                        
        detectAndDisplay(frame)
        
        
        
        
        #-------------------------------------------------------------------------------------------

        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        #colorBackgroundText(frame,  f'{categories[y]}', FONTS, 0.7, (30,250),2, PINK, YELLOW)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            colorBackgroundText(frame,  f'{categories[y]}', FONTS, 1.7, (int(frame_height/2), 160), 2, YELLOW, pad_x=6, pad_y=6)
            v_end=time.time()
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            #colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, PINK, YELLOW)
            if ratio >4.5:
                CEF_COUNTER +=1
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    c_start = time.time()
                    COUNTER+=1
                    colorBackgroundText(frame,  f'CLOSED', FONTS, 1.7, (int(frame_height/2), 100), 2, YELLOW, pad_x=6, pad_y=6)
                    if closed_time>6:
                        colorBackgroundText(frame,  f'SLEEP:(', FONTS, 1.7, (int(frame_height/2), 100), 2, YELLOW, pad_x=6, pad_y=6)
                        
              
            elif ratio<= 4.5:
                dCOUNTER+=1
                c_end=time.time()
                if dCOUNTER>60:
                    CEF_COUNTER=0
                    dCOUNTER=0
#                 elif CEF_COUNTER>CLOSED_EYES_FRAME and dCOUNTER <=60:
#                     colorBackgroundText(frame,  f'KEEP FOCUS', FONTS, 1.7, (int(frame_height/2), 100), 2, YELLOW, pad_x=6, pad_y=6)
                    
            if c_start>c_end:
                closed_time1=math.trunc(c_start-c_end)
                if closed_time1!=closed_time:
                    closed_time=closed_time1
                    if closed_time>=6:
                        sleep+=1
                        
#             #colorBackgroundText(frame,  f'COUNTER: {COUNTER}', FONTS, 0.7, (30,150),2)
            #cv2.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, GREEN, 1, cv2.LINE_AA)
            #cv2.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, GREEN, 1, cv2.LINE_AA)
        
        else:
            v_start=time.time()
            #colorBackgroundText(frame,  f'{categories[y]}', FONTS, 1.7, (int(frame_height/2), 160), 2, YELLOW, pad_x=6, pad_y=6)
                
        if v_start> v_end:
            vanish1=math.trunc(v_start-v_end)
            if vanish!=vanish1:
                vanish=vanish1
                #vanished over 30 secs _>count GONE
                if vanish==30:
                    GONE+=1
        colorBackgroundText(frame,  f'Close_Time : {closed_time}', FONTS, 0.9, (30,50),2, PINK, YELLOW)   
        colorBackgroundText(frame,  f'Sleep : {sleep}', FONTS, 0.9, (30,100),2, PINK, YELLOW)          
        
        colorBackgroundText(frame,  f'vanish_time :{vanish}', FONTS, 0.9, (30,150),2, PINK, YELLOW)
        colorBackgroundText(frame,  f'GONE :{GONE}', FONTS, 0.9, (30,200),2, PINK, YELLOW)
        
        
        
        ##hand setting
        img = cv2.flip(frame, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                

                # Draw gesture result
                if idx in study_gesture.keys():
                    colorBackgroundText(frame, study_gesture[idx].upper(), FONTS, 0.9, (30,300),2, PINK, YELLOW)
#                     cv2.putText(frame, text=study_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
    
#         end_time = time.time()-start_time
#         fps = frame_counter/end_time
#         frame =textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key==ord('q') or key ==ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()


# In[ ]:





# In[ ]:




