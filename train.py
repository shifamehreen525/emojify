#importing packages
import numpy as np
import tensorflow
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


# work with data
train_dir = 'data/train'
val_dir = 'data/test'

#ImageDataGenrator --> generate image from the data files in our directory
# pixels of image --> array --> scale 1 to 255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


#generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48), # 48*48 pixels
        batch_size=64, # no of training samples utilized in one iteration
        color_mode="grayscale", #black and white images
        class_mode='categorical') #categories
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


#create model
emotion_model = Sequential()  #seuquential model --> layer by layer model --> single i/p and o/p


#adding layers
#CNN --> used popularly for image data 
#feature map --> i/p or patterns repeat and maps it to particular feature
#2d cnn --> filters, height and width of cnn, activation function = relu, (pixels,grey)
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#down sampling --> training on low subset of samples
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
#dropping neurons --> to prevent overfitting
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#max pooling
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
#flatten
emotion_model.add(Flatten())
#dense --> hidden layer, 1024 --> neurons
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
#7 categories ==> 1024 units into 7 
emotion_model.add(Dense(7, activation='softmax'))



#compile model  ---> loss: degree of error, optimizer, metric to track --> accuracy
emotion_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.0001, decay=1e-6),
                      metrics=['accuracy'])

#train model --> calulate weights
# fit it to the model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64, # since, batch size --> 64
        epochs=60,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

#save model --> save weights
emotion_model.save_weights('model.h5')  #saved only weights not entire model

