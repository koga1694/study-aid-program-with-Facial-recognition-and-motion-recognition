#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
import shutil
import pathlib
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras import optimizers
import tensorflow as tf
# from keras import backend as K
import matplotlib.pyplot as plt
import PIL
import numpy as np


# In[ ]:


import shutil
import pathlib
import tensorflow as tf

data_dir = pathlib.Path(r'C:\workplace\cnn_daicon\train')
print(len(list(data_dir.glob(r'*\*.jpg'))))


batch_size = 16
num_epochs = 50
img_height = 244
img_width = 244

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 97,
    shuffle=True,
    image_size = (img_height, img_width),
    batch_size = batch_size)
    
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 97,
    shuffle=True,
    image_size = (img_height, img_width),
    batch_size = batch_size)

class_names = train_ds.class_names
print(class_names)
print(train_ds)


# In[ ]:


# Model
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(244,244,3))
x = net.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(3, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
# for layer in net_final.layers[:freeze_layers]:
#     layer.trainable = False
# for layer in net_final.layers[freeze_layers:]:
#     layer.trainable = True
for layer in net_final.layers:
    layer.trainable = True
net_final.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = 'inception_resnet_{epoch:02d}-{val_loss:.2f}.hdf5',
                                                      monitor = 'val_loss',
                                                      save_best_only = True, verbose = 1)
print(net_final.summary())


# In[ ]:


# Model train
history = net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // batch_size,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // batch_size,
                        epochs = num_epochs,
                        callbacks = [early_stopping, model_checkpoint])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




