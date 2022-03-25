#!/usr/bin/env python
# coding: utf-8

# In[54]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import PIL
from PIL import Image 
import os, shutil
import glob
get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


# understand, not_understand, netural 폴더로 이미지 이동

os.mkdir('./datasample')
os.mkdir('./datasample/not_understand')
os.mkdir('./datasample/understand')
os.mkdir('./datasample/neutral')

count = 0 

for root, subdirs, files in os.walk(r'./원천데이터'): 
    # fear, sad, angry, disgust - not understand
    # netural - netural
    # happy - understand
    # suprise - suprise (보류)
    count += 1
    print(count)
    
    for f in files:
        if '기쁨' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./\datasample\understand')
    
    
        elif '당황' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./\datasample\not_understand')
    
    
        elif '불안' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./\datasample\not_understand')
    
    
        elif '상처' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./datasample\not_understand')
            
    
        elif '분노' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./\datasample\not_understand')
            
    
        elif '슬픔' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./\datasample\not_understand')
            
    
        elif '중립' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'./\datasample\neutral')



# In[56]:



path=r'./\datasample'
understand = glob.glob(path+r'\understand'+'\*')
not_understand = glob.glob(path+r'\not_understand'+'\*')
neutral = glob.glob(path+r'\neutral'+'\*')

print(len(understand))
print(len(not_understand))
print(len(neutral))

def rename(files):
    if 'understand' in files[0] and 'not' not in files[0]:
        for index, file in enumerate(files):
            os.rename(file, os.path.join(path+r'\understand', 'understand_'+f'{index}.jpg'))
        understand = glob.glob(path+r'\understand'+'\*')
        print(f'understand {index}번째 변경 완료')
        
    elif 'not_understand'  in files[0]:
        for index, file in enumerate(files):
            os.rename(file, os.path.join(path+r'\not_understand', 'not_understand_'+f'{index}.jpg'))
        not_understand = glob.glob(path+r'\not_understand'+'\*')
        print(f'not_understand {index}번째 변경 완료')
        
    elif 'neutral' in files[0]:
        for index, file in enumerate(files):
            os.rename(file, os.path.join(path+r'\neutral', 'neutral_'+f'{index}.jpg'))
        netural = glob.glob(path+r'\neutral'+'\*')
        print(f'neutral {index}번째 변경 완료')

rename(understand)
rename(not_understand)
rename(neutral)


# In[57]:


import natsort
failed_list=[]

def nat(num):
    if num<0:
        return 0
    return int(num)


def cropped(frame):
    for i in frame: 
        detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt" , "res10_300x300_ssd_iter_140000.caffemodel")
        print(i)
        image = cv2.imread(i)
        base_img=image.copy()
        
        original_size = base_img.shape
        target_size = (128,128)
        image = cv2.resize(image, target_size)
        

        imageBlob = cv2.dnn.blobFromImage(image = image)

        detector.setInput(imageBlob)
        detections = detector.forward()
        column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
        detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
        detections_df = detections_df[detections_df['is_face'] == 1]
        detections_df = detections_df[detections_df['confidence'] >= 0.50]
        
        if len(detections_df.index) == 0:
            failed_list.append(i)        
            continue
        
        
        for k, instance in detections_df.iterrows():
            
            detected_face = base_img[nat(instance["top"] * original_size[0] ):nat(instance["bottom"] * original_size[0]),nat(instance["left"] * original_size[1]):nat(instance["right"] * original_size[1])]
            #detected_face = image[int(top):int(bottom), int(left):int(right)]
            #cv2.cvtColor(detected_face)
            print(instance["top"] * original_size[0],instance["bottom"] * original_size[0],instance["left"] * original_size[1],instance["right"] * original_size[1])
            
            cv2.imwrite(i,detected_face)
            #plt.imshow(detected_face)
            #plt.axis('off')
            #plt.show() 

cropped(natsort.natsorted(glob.glob(r'./\datasample\understand\*')))
cropped(natsort.natsorted(glob.glob(r'./\datasample\not_understand\*')))
cropped(natsort.natsorted(glob.glob(r'./\datasample\neutral\*')))


# In[58]:


failed_list


# In[59]:


# crop 안된 파일 삭제
for i in failed_list:
    os.remove(i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




