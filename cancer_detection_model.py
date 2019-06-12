import numpy as np
import tensorflow
import keras
from keras.models import Sequential    
cnn_model=Sequential()


from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

x_train=train_datagen.flow_from_directory(r"D:\AAA\COURSES\AI\PROJECT\data",target_size=(64,64),batch_size=32,class_mode='binary')    

# path of training data:D:\AAA\COURSES\AI\PROJECT\data

#input layer
cnn_model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())

#hidden layer
cnn_model.add(Dense(128,activation='relu'))
cnn_model.add(Dense(128,activation='relu'))
cnn_model.add(Dense(128,activation='relu'))
cnn_model.add(Dense(128,activation='relu'))

#output layer
cnn_model.add(Dense(1,activation='sigmoid'))
##########
#training
cnn_model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
cnn_model.fit_generator(x_train,samples_per_epoch=10000,epochs=20,nb_val_samples=100)
##########

###########
cnn_model.save('cancer_classification_model.h5')


