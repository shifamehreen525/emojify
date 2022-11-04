import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading

#recreated model
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

emotion_model.load_weights('model.h5')  # loading weights along with the same model

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "  Disgust  ", 2: "  Fear ", 3: "   Happy   ", 
4: "  Neutral  ", 5: "    Sad    ", 6: "  Surprised  "}
#import avatars 
# cur_path = os.path.dirname(os.path.abspath(__file__)) 
emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgust.png",2:"./emojis/fear.png",
3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surprised.png"}


global last_frame1 
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
global show_text
cap1 = cv2.VideoCapture(0)   #or specify recorded video inside brackets 
show_text=[0]
global frame_number 


def show_subject():      
    # cv2.imshow('Video', frame1)

    if not cap1.isOpened():                             
        print("Can't open the camera!")

    # global frame_number
    # length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) #length of frame --> to know when to exit
    # frame_number += 1
    # if frame_number >= length:
    #     exit()
    # cap1.set(1, frame_number)

    flag1, frame1 = cap1.read() #read frame by frame --> frame: array, flag: read anything or not
    frame1 = cv2.resize(frame1,(600,500))
    bounding_box = cv2.CascadeClassifier(r"C:\Users\shifa\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #color to grayscale conversion
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        #refine image
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        #resize
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        #predicting
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+120, y+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex


    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)  #color conversion
        img = Image.fromarray(pic) #array to image
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update() #update main thread
        lmain.after(10, show_subject) #after 10 ms calls function again
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     exit()


def show_avatar():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    # img3 = img2.resize(1500, 1500)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',35,'bold'))
    
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)



if __name__ == '__main__':
    # frame_number = 0 #inc frame by frame
    root=tk.Tk()   

    # img = ImageTk.PhotoImage(Image.open("logo.png"))
    # heading = Label(root,image=img,bg='black')
    # heading.pack() 

    # heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
    # heading2.pack()

    lmain = tk.Label(master=root,padx=50,bd=10) #contain our video
    lmain2 = tk.Label(master=root,bd=10) #contain our image
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black') #quit button
    lmain.pack(side=LEFT)
    lmain.place(x=30,y=100)
    lmain3.pack()
    lmain3.place(x=900,y=-10)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=90)
    
    root.title("Photo To Emoji")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)

    #run continuously
    #create new thread
    #Python threading --> allows you to have different parts of your program run concurrently 
    # and can simplify your design
    threading.Thread(target=show_subject).start()  #our video
    threading.Thread(target=show_avatar).start()   #our emoji
    root.mainloop() #while loop that never ends
    cap1.release()
