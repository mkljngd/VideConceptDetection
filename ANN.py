# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:21:30 2019

@author: Ad
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


train = pd.read_csv('lbpdata_train.csv',header=None)
train.columns=['1','2','3','4','5','6','7','8','9','10','Class']
X_train=train.iloc[:,:10].values
Y_train=train.iloc[:,10].values
test_ = pd.read_csv('lbpdata_test.csv',header=None)
test=test_.iloc[:,:10]
test.columns=['1','2','3','4','5','6','7','8','9','10']
X_test=test

no_of_inputs=10
no_of_output_classes=5
no_of_hidden_layer_nodes=math.ceil((no_of_inputs+no_of_output_classes)/2)



onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(train).toarray()
XTrain = X[:, 5:]
YTrain=X[:,:5]

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))


classifier=Sequential()

classifier.add(Dense(output_dim = no_of_hidden_layer_nodes, init = 'uniform', activation = 'relu', input_dim = no_of_inputs))

classifier.add(Dense(output_dim = no_of_hidden_layer_nodes, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = no_of_hidden_layer_nodes, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = no_of_output_classes, init = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(XTrain, YTrain, batch_size = 10, nb_epoch = 10000)

classifier.save('model.h5')

y_class=[]
y_pred = classifier.predict(X_test)

for x in range(0,y_pred.shape[0]):
    row=[]
    clas=[]
    maxm=max(y_pred[x])
    for idx,i in enumerate(y_pred[x]):
        row.append(i)
        if(i==maxm):
            clas.append(idx+1)
    row=row+clas
    
    y_class.append(row)

np.savetxt('predictions.csv',y_class,delimiter=",")
