# Import Train Modules.
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Data preparation and preprocessing

train_dir = 'data/train' # Path to the training and test data
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale", # grayscale images for faster processing and less memory usage
        class_mode='categorical') # categorical labels for emotions

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Build the CNN model (Detection Model)

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# compile and train model

cv2.ocl.setUseOpenCL(False) # disable OpenCL usage in OpenCV

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# compile and train model with Adam optimizer
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)
emotion_model.save_weights('emotion_model.h5')

# Real time emotion detection
cap = cv2.VideoCapture(0)
while True: # loop to capture video frames
    ret, frame = cap.read() # read the frame
    if not ret: # if the frame is not read
        break
    bounding_box = cv2.CascadeClassifier('/') #
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces: # for each face detected
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # draw a rectangle around the face
        roi_gray_frame = gray_frame[y:y + h, x:x + w] # region of interest in the frame
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0) # crop image
        emotion_prediction = emotion_model.predict(cropped_img) # predict the emotion
        maxindex = int(np.argmax(emotion_prediction)) # get the index of the emotion with the highest probability
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC)) # show the video frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # if 'q' is pressed, quit. Else, continue
        break

cap.release()
cv2.destroyAllWindows()