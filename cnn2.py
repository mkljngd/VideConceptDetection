# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:46:55 2019

@author: Ad
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.preprocessing import image
from keras.optimizers import SGD,Adam
from keras.regularizers import l2

import numpy as np
import os

class_list=['Archery','BaseballPitch','CricketBowling','CricketShot','Kayaking']
d={'Archery':1,'BaseballPitch':2,'CricketShot':3,'CricketBowling':4,'Kayaking':5}
def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3,3), input_shape = (32, 32, 3),padding='valid',activation='relu'))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu',kernel_regularizer=l2(0.0001)))
#classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = SGD(lr=0.001,nesterov=True), loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""
train_datagen = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                rescale=1/255.,
                                fill_mode='nearest',
                                channel_shift_range=0.2*255)
"""
test_datagen = ImageDataGenerator(rescale = 1./255)
directory='./test_data'

test=ImageDataGenerator(rescale=1).flow_from_directory(directory, target_size=(32, 32), class_mode='categorical', batch_size=32, shuffle=True,  interpolation='nearest')

training_set = train_datagen.flow_from_directory('./train_data',
                                                 target_size = (32, 32),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./test_data',
                                            target_size = (32, 32),
                                            batch_size = 10,
                                            class_mode = 'categorical')


history=classifier.fit_generator(training_set,
                         samples_per_epoch = 14000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 6000)


pred=classifier.predict_generator(test_set,steps=1000)
classifier.save('25_epoch_model.h5')
classifier.save_weights('25_epochs.h5')


from keras.models import load_model
from keras.preprocessing import image


# image folder
folder_path = './test_data/'

# path to model

# dimensions of images
img_width, img_height = 32, 32

# load the trained model
model = load_model('./25_epoch_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.0001),
              metrics=['accuracy'])

# load all images into a list\
res_pred=[]
for classes in class_list:
    res=[]
    for imgs in os.listdir(folder_path+classes):
        print(imgs)
        test_image = image.load_img('./test_data/'+classes+'/'+imgs, target_size = (32, 32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        res.append(test_image)
    
    # stack up images list to pass for prediction
    images = np.vstack(res)
    class_pred = model.predict_classes(images, batch_size=10)
    res_pred=res_pred+np.ndarray.tolist(class_pred)
    
    print(class_pred)
for i in range(len(res_pred)):
    res_pred[i]=res_pred[i]+1
    
np.savetxt('predictANNlol.csv',res_pred,delimiter=',')
