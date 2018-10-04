import csv
import cv2
import numpy as np
from sklearn.model_selection import validation_split
from sklearn.utils import shuffle
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import 
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt


lines = []

model = Sequential()
model.add(Lambda(lambda: x: x/255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Flatten(input_shape = (160, 320, 3)))
#model.add(Dense(1))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle = True,nb_epochs)

model.save('model.h5')
