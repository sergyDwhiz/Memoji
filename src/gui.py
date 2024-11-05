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

'''
Here, we define the labels for the emotions that the model can detect, as well as
disable OpenCL usage in OpenCV
'''
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}
emoji_dist = {0: 'emojis/angry.png', 1: 'emojis/disgusted.png', 2: 'emojis/fearful.png', 3: 'emojis/happy.png', 4: 'emojis/neutral.png', 5: 'emojis/sad.png', 6: 'emojis/surpriced.png'}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype = np.uint8) # initializing the frame
global cap1 # video capture
show_text = [0]  # text to show on screen


'''
Here, we capture and process Video Frames
'''
def show_vid():
    cap1 = cv2.VideoCapture(0) # capturing video from the camera
    if not cap1.isOpened(): # if the camera is not opened
        print('Error! Check your camera')
    flag1, frame1 = cap1.read() # reading the frame
    frame1 = cv2.resize(frame1, (600, 500))  # resizing the frame

    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) # converting the frame to grayscale
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5) # detecting faces in the frame

    for (x, y, w, h) in num_faces: # for each face detected
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # draw a rectangle around the face
        roi_gray_frame = gray_frame[y:y +h, x:x + w] # region of interest in the frame
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0) # crop image
        prediction = emotion_model.predict(cropped_img)

        max_index = int(np.argmax(prediction)) # get the index of the emotion with the highest probability
        show_text[0] = max_index # set the text to show on screen

    if flag1 is None: # if the frame is not captured
        print('Major Error!')
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB) # convert the frame to RGB
        img = Image.fromarray(pic) # convert the frame to an image
        imgtk = ImageTk.PhotoImage(image = img) # convert the image to a Tkinter image
        lmain.imgtk = imgtk # set the image to the Tkinter image
        lmain.configure(image = imgtk) # configure the image
        lmain.after(10, show_vid) # call the function after 10ms
    if cv2.waitKey(1) & 0xFF == ord('q'): # if 'q' (FOr Quit) is pressed
        exit()

'''
Now, we need to display the corresponding Emoji
'''

def show_vid2():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) # convert the frame to RGB
    img2 = Image.fromarray(frame2) # convert the frame to an image
    imgtk2 = ImageTk.PhotoImage(image = img2) # convert the image to a Tkinter image
    lmain2.imgtk2 = imgtk2 # set the image to the Tkinter image
    lmain2.configure(image = imgtk2) # configure the image

    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold')) # edit text
    lmain2.after(10, show_vid2) # call the function after 10ms


# Set up the GUI

if __name__ == '__main__':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    heading = Label(root, image=img, bg='black')

    heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')

    heading2.pack()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)

    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()