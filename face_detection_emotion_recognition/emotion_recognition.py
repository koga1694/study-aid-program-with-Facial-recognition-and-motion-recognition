#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import PIL
from PIL import Image 

model_name='res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name='deploy.prototxt.txt'
model77=tf.keras.models.load_model('last_model.hdf5')
categories=["netural","not_understand","understand"]


# In[ ]:


def detectAndDisplay(frame):
    detector=cv2.dnn.readNetFromCaffe(prototxt_name,model_name)
    original_size = frame.shape
    target_size = (300,300)
    
    
    image = cv2.resize(frame, target_size)
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

        detected_face = frame[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

        #감정값 예측하기

        emo_data=Image.fromarray(detected_face[:,:,::-1])
        emo=np.asarray(emo_data)
        emo=cv2.resize(emo, (299,299))
        emo=emo.reshape((-1, 299,299,3))
        test = model77.predict(emo)
        y=model77.predict(emo).argmax()
        print(y)
        print(test)
        emo_list = np.append(emo_list, y)

        #####
        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:


            cv2.putText(frame, f'{categories[y]}', (int(left*aspect_ratio_x+100), int(top*aspect_ratio_y+250)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f'{round(np.max(test) * 100)}%', (int(left*aspect_ratio_x+120), int(top*aspect_ratio_y+280)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            
            
            
#             cv2.rectangle(frame, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), 
#                           (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image

    cv2.imshow("Face detection by Dnn",frame)


# In[ ]:


# Cam 실행
cap=cv2.VideoCapture(0)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

