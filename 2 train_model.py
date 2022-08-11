# python train_model.py data.csv 5000 

import sys
import pickle
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D

data = pd.read_csv(sys.argv[1], index_col=0)
labels = np.array(data.labels)
imagePaths = np.array(data.images)

# One hot encoding for labels 
lb = LabelBinarizer().fit(labels)
Y = lb.transform(labels)

# Saving model for decoding
pickle.dump(lb, open('labelsLB.dat', 'wb'))

# Loading images
images = []
for path in imagePaths:
    images.append(imread(path, as_gray=True))
images = np.array(images)

# Adding third dimension for keras
X = np.expand_dims(images, axis=3)

# Image augmentation parameters
datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=10,
    width_shift_range=0.3, 
    height_shift_range=0.3)

# Building neural network
model = Sequential()

# First convolutional layer w/ max pooling
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D())

# Second convolutional layer w/ max pooling
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D())

# Hidden layers with 1024
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.33))

# Output layer with 94 nodes
model.add(Dense(94, activation="softmax"))

# Build the TensorFlow model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model
model.fit(datagen.flow(X, Y, batch_size=97), validation_data=(X, Y), epochs=sys.argv[2], verbose=1)

# Saving the model
model.save('model1')