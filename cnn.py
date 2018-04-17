# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:37:52 2018

@author: saurabh
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
import os

os.getcwd()

os.chdir("C:\\Users\\saurabh\\Desktop\\New folder (2)")

classifier=Sequential()

classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# second layer

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory("training",
target_size=(64,64),
batch_size=2,
class_mode='binary') 

testing_set=test_datagen.flow_from_directory("testing",
target_size=(64,64),
batch_size=2,
class_mode='binary')                                              

classifier.fit_generator(training_set,
steps_per_epoch=5,
epochs=5, 
validation_data=testing_set,
validation_steps=20) 

#making new prediction

import numpy as np
from keras.preprocessing import image

test_image=image.load_img("single_prediction\\cat_or_dog.jpg",target_size=(64,64))
                       
test_image=image.img_to_array(test_image) 

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)

training_set.class_indices

if result [0] [0] == 1:
   prediction = 'dog'
else:
   prediction = 'cat'                       
                         
                                 