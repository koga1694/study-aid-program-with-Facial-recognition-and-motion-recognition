#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# understand, not_understand, netural, su(surprise) 폴더로 이미지 이동

import os, shutil
count = 0 
for root, subdirs, files in os.walk(r'C:\workplace\FP\test'): 
    # fear, sad, angry, disgust - not understand
    # netural - netural
    # happy - understand
    # suprise - suprise (보류)
    count += 1
    print(count)
    
    for f in files:
        if 'netural' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\netural')
    
    
        elif 'surprise' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\su')
    
    
        elif 'happy' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\understand')
    
    
        elif 'disgust' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\not_understand')
            
    
        elif 'angry' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\not_understand')
            
    
        elif 'fear' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\not_understand')
            
    
        elif 'sad' in f:
            file_to_move = os.path.join(root, f)
            shutil.move(file_to_move, r'C:\workplace\FP\face\not_understand')


# In[ ]:


# 이미지 이름 변경

import glob
import os

path = r'C:\workplace\FP\face'
understand = glob.glob(path+r'\understand'+'\*')
not_understand = glob.glob(path+r'\not_understand'+'\*')
netural = glob.glob(path+r'\netural'+'\*')
su = glob.glob(path+r'\su'+'\*')

print(len(understand))
print(len(not_understand))
print(len(netural))

def rename(files):
    if 'understand' in files[0]:
        for index, file in enumerate(files):
            os.rename(file, os.path.join(path+r'\understand', 'understand_'+f'{index}.jpg'))
        understand = glob.glob(path+r'\understand'+'\*')
        print(f'understand {index}번째 변경 완료')
        
    elif 'not_understand' in files[0]:
        for index, file in enumerate(files):
            os.rename(file, os.path.join(path+r'\not_understand', 'not_understand_'+f'{index}.jpg'))
        not_understand = glob.glob(path+r'\not_understand'+'\*')
        print(f'not_understand {index}번째 변경 완료')
        
    elif 'netural' in files[0]:
        for index, file in enumerate(files):
            os.rename(file, os.path.join(path+r'\netural', 'netural_'+f'{index}.jpg'))
        netural = glob.glob(path+r'\netural'+'\*')
        print(f'netural {index}번째 변경 완료')
        
    
        
# rename(understand)
# rename(not_understand)
# rename(netural)

