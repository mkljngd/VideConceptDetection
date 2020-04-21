# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:46:55 2019

@author: Aditya Vartak
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
classifier.add(Convolution2D(16, (3,3), input_shape = (32, 32, 3),padding='valid'))
classifier.add(BatchNormalization())
classifier.add(Activation('softmax'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, (3,3), input_shape = (32, 32, 3),padding='valid'))
classifier.add(BatchNormalization())
classifier.add(Activation('softmax'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, (3,3), input_shape = (32, 32, 3),padding='valid'))
classifier.add(BatchNormalization())
classifier.add(Activation('softmax'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('train_data',
                                                 target_size = (32, 32),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_data',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            class_mode = 'categorical')


history=classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 2000)
class_res=[]
for classes in class_list:
    res=[]
    for imgs in os.listdir('test_data/'+classes):
        print(imgs)
        test_image = image.load_img('test_data/'+classes+'/'+imgs, target_size = (32, 32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        print(result)
        arr=[i for i in np.nditer(result)]
        max_arr=max(arr)
        re = np.where(arr == np.amax(arr))
        arr.append(re[0]+1)
        class_res.append(arr)



np.savetxt('predictCNN.csv',class_res,delimiter=',')
        
