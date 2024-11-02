import tkinter as tk      # GUI
from tkinter import *
import cv2    # image processing and Video Capture
from PIL import Image, ImageTk
import numpy as np   # numerical operations
import os
from keras.models import Sequential # building and loading model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.proprocessing.image import ImageDataGenerator

'''
This section defines a CNN for emotion detection, consisting of 3 Convolutional layers, 2 MaxPooling layers,
2 Dropout layers, 2 Dense layers and 1 Flatten layer.
The model is then loaded with the weights from the trained model.
'''
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation = 'relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation = 'softmax'))
emotion_model.load_weights('model.h5')
