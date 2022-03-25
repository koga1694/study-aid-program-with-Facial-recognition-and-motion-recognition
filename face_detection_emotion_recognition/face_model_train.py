#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
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
import matplotlib.pyplot as plt
import PIL
import numpy as np


# In[ ]:


# Data preprocessing
import shutil
import pathlib
import tensorflow as tf

data_dir = pathlib.Path(r'C:\workplace\cnn_daicon\train')
print(len(list(data_dir.glob(r'*\*.jpg'))))


batch_size = 16
num_epochs = 50
img_height = 299
img_width = 299

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


# model train

net = InceptionResNetV2(include_top=False, weights=None, input_tensor=None,
               input_shape=(299,299,3))



inputs = keras.Input(shape=(299,299,3))
x = (inputs)
x = preprocess_input(x)
x = net(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(class_names), activation='softmax', name='softmax')(x)

net_final = Model(inputs=inputs, outputs=output_layer)

# freeze layer
# freeze_layers = 4
# for layer in net_final.layers[:freeze_layers]:
#     layer.trainable = False
# for layer in net_final.layers[freeze_layers:]:
#     layer.trainable = True

for layer in net_final.layers:
    layer.trainable = True
    
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.metrics.SparseCategoricalAccuracy()
net_final.compile(optimizer=Adam(lr=1e-4),
                  loss=loss, metrics=metrics)

# 콜백
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 6)
model_checkpoint = ModelCheckpoint(filepath = 'inception_resnet_{epoch:02d}-{val_loss:.2f}.hdf5',
                                                      monitor = 'val_loss',
                                                      save_best_only = True, verbose = 1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr = 1e-6
    )


print(net_final.summary())


# In[ ]:


history = net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // batch_size,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // batch_size,
                        epochs = num_epochs,
                        callbacks = [early_stopping, model_checkpoint, reduce_lr])


# In[3]:


jupyter nbconvert --to script face_model_train.ipynb


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




