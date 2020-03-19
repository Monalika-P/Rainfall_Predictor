# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:38:54 2020

@author: MONALIKA P

"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
df= pd.read_csv('seattleWeather_1948-2017.csv')

df['rain']=[1 if i==True else 0 for i in df['RAIN']]

df.dropna(inplace=True)

X = df.iloc[:, 1:4].values
y = df.iloc[:, -1].values

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

ann  = Sequential()
ann.add(Dense(units= 256, kernel_initializer= 'uniform', activation = 'relu', input_dim=3))
ann.add(Dense(units= 512, kernel_initializer= 'uniform', activation = 'relu'))
ann.add(Dense(units= 1024, kernel_initializer= 'uniform', activation = 'relu'))
ann.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the model
from tensorflow.keras.optimizers import RMSprop
ann.compile(optimizer = RMSprop(lr = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])

#Building the checkpoint
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') >= 0.99):
            print("Reached 99% of accuracy so Stopped Training!")
        
callback = mycallback()        
        
#Fitting the model
ann.fit(X_train, y_train, batch_size = 100, epochs = 100, callbacks = [callback])

#Predicting the test set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)

#Making Single prediction
# Enter data in order 
new_pred = ann.predict(sc.transform(np.array([[0.5, 43, 50]])))
new_pred = (new_pred > 0.5)
