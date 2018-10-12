import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model
import matplotlib
import matplotlib.pyplot as plt

path = 'My_Data/driving_log.csv'
#lines = []

#with open(path) as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

samples = []

with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)        
        
        
images = []
measurements = []
#for line in samples:
#    source_path = line[0]
#    filename = source_path.split('/')[-1]
#    current_path = 'My_Data/IMG/' + filename
#    image = cv2.imread(current_path)
#    images.append(image)
#    measurement = float(line[3])
#    measurements.append(measurement)
    
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'My_Data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format


#X_train = np.array(images)
#y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(80, 320, 3)))
#model.add(Flatten(input_shape=(160, 320, 3)))
#model.add(Dense(1))
#model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
#model.fit_generator(train_generator,validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
model.save('model.h5')

#history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
