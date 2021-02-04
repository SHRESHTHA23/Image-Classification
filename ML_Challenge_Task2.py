#!/usr/bin/env python
# coding: utf-8

# # Task 2

# ### Importing Required Packages

# In[ ]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage.interpolation import shift
import random
import cv2
from PIL import Image
import matplotlib
import csv


# ### Importing Required Packages for CNN Classifier

# In[ ]:


from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ### Loading the Data for training our Model

# In[ ]:


train_data= np.load('training-dataset.npz')


# In[ ]:


feature = train_data['x']
label = train_data['y']


# In[ ]:


print(feature.shape)
print('-----')
print(label.shape)


# In[ ]:


#This is to reshape our training data

feature = feature.reshape(124800, 28, 28)
print(feature.shape)


# In[ ]:


np.unique(label)


# ### Splitting our Dataset into Train, Validation and Test Set using Stratification

# In[ ]:


train_data, val_data, train_label, val_label = train_test_split(feature,label,stratify = label,test_size = 0.3,random_state = 640)


# In[ ]:


val_data, test_data, val_label, test_label = train_test_split(val_data,val_label,stratify = val_label,test_size = 0.5,random_state = 640)


# ### Training our Model by adding Noise
# 
# #### Creating a Function for adding noise to our train images

# In[ ]:


def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255 
            else:
                output[i][j] = image[i][j]
    return output
    
#As sourced from "https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv"


# In[ ]:


noise_train = []
for i in range(0,len(train_data)):
    x = sp_noise(train_data[i], 0.1)
    noise_train.append(x)
train_data = np.array(noise_train)

noise_val = []
for i in range(0,len(val_data)):
    x=sp_noise(val_data[i], 0.1)
    noise_val.append(x)
val_data = np.array(noise_val)

noise_test = []
for i in range(0,len(test_data)):
    x=sp_noise(test_data[i], 0.1)
    noise_test.append(x)
test_data = np.array(noise_test)


# In[ ]:


print(train_data.shape)
print()
print(val_data.shape)
print()
print(test_data.shape)


# In[ ]:


plt.imshow(train_data[0])


# ### CNN Classifier

# In[ ]:


# Converting our Labels to categorical

final_train_label = to_categorical(np.array(train_label), num_classes = 27)
final_val_label = to_categorical(np.array(val_label), num_classes = 27)


# In[ ]:


#Scaling our Dataset

train_data_scaled = train_data / 255
val_data_scaled = val_data / 255
test_data_scaled = test_data / 255


# In[ ]:


img_rows,img_cols = 28,28


# In[ ]:


final_train_data = train_data_scaled.reshape(train_data.shape[0],img_rows,img_cols,1)
final_val_data = val_data_scaled.reshape(val_data.shape[0],img_rows,img_cols,1)
final_test_data = test_data_scaled.reshape(test_data.shape[0],img_rows,img_cols,1)


# #### Building CNN Model

# In[ ]:


# Instantiate model
model = Sequential()

# Convolution layer 1
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Convolution layer 2
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Pooling layer
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.25))

# Convolution layer 3
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Convolution layer 4
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Flattening image
model.add(Flatten())

# Dense layer
model.add(Dense(256, activation='relu'))

# Output layer
model.add(Dense(27, activation='softmax'))

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# This is to prevent Overfitting of our Model by using patience(Epochs) of 3

early_stop = EarlyStopping(monitor='val_loss', patience=3)


# In[ ]:


image_gen = ImageDataGenerator(rotation_range=10,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              fill_mode='nearest')


# In[ ]:


model.fit_generator(image_gen.flow(final_train_data, final_train_label), epochs=10, validation_data=image_gen.flow(final_val_data, final_val_label), callbacks=[early_stop])


# In[ ]:


CNN_Metrics = pd.DataFrame(model.history.history)
CNN_Metrics


# In[ ]:


#Accuracy on our Test split(unseen data) from the Training Data

y_pred = model.predict_classes(final_test_data)
print(classification_report(test_label, y_pred))


# #### Count of Incorrect Predictions in Test Split(unseen data) from the Traning Data

# In[ ]:


count = 0
for i in range(len(y_pred)):
    if y_pred[i] != test_label[i]:
        count += 1
print((count))


# ### Loading the Test Dataset

# In[ ]:


test_Data = np.load('test-dataset.npy')


# In[ ]:


print(test_Data.shape)


# In[ ]:


true_letter = test_Data[615]
#true_letter = true_letter.reshape(30,28)
plt.imshow(true_letter)


# #### Splitting our Test Data Images into subset of (28,28) Images

# In[ ]:


new_test_data = []
for i in range(0,len(test_Data)):
    true_letter = test_Data[i]
    M = true_letter.shape[0] - 2 
    N = (true_letter.shape[1] - 28)//5
    t_l = [true_letter[x:x+M,y:y+N] for x in range(0,true_letter.shape[0],M) for y in range(0,true_letter.shape[1],N)]
    new_test_data.append(t_l)
final_test_data = np.array(new_test_data)

plt.imshow(final_test_data[0][0])


# #### Getting the Top 5 Accuracy for Each Image in our Test Data

# In[ ]:


#This code is for getting the Top 5 Accuracy.

final_five_list = []
for i in range(0,len(final_test_data)):
    best_5_letter = []
    for j in range(0,5):
        #The below code will give five prediction for each subset image based on their probability(confidence scores)
        y_pred = model.predict_proba(final_test_data[i][j].reshape(1,28,28,1))
        best_5 = np.argsort(y_pred,axis = 1)[:,-5:]
        best_5 = best_5.tolist() 
        best_5_letter.append(best_5)
    best_5_letter = np.array(best_5_letter)
    best_5_letter = np.flip(best_5_letter, axis = 2) 
    best_5_letter = best_5_letter.T
    final_five_list.append(best_5_letter)
final_five_list = np.array(final_five_list).reshape(len(final_test_data), 25)


# #### Converting the final Output into the desired form as per Requirement

# In[ ]:


final_5 = final_five_list.tolist()
Top_5_Accuracy = []
for x in final_5:
    value = []
    for y in range(0,len(x),5):
        z = x[y:y+5]
        z1 = []
        for k in z:
            k = str(k)
            k = k.zfill(2) #to add leading zeroes to the labels
            z1.append(k)
        z1  = "".join(z1) #joining all the predictions of 1 image together as a string
        value.append(z1)
    Top_5_Accuracy.append(value)
Top_5_Accuracy = np.array(Top_5_Accuracy)


# #### Saving the final output to a CSV file.

# In[ ]:


pd.DataFrame(Top_5_Accuracy).to_csv("Top_5_Accuracy_Group45.csv", header = None, index = None)


# # Additional Content

# #### The below code can be used for getting only the Single prediction for each of the Test Data Images

# In[ ]:


#final_y_pred = []
#for i in range(0,len(final_test_data)):
#    y_pred_new = []   for j in range(0,5):
#        y_pred = model.predict_classes(final_test_data[i][j].reshape(1,28,28,1))
#        y_pred = str(y_pred[0]).zfill(2)
#        y_pred_new.append(y_pred)
#    y_pred_new = "".join(y_pred_new)
#    final_y_pred.append(y_pred_new)
#final_y_pred = np.array(final_y_pred)
#print(final_y_pred)


# #### The below code can be used if we want to remove Noise from our Test Images and then use the model trained in Task 1 for the predictions

# In[ ]:


# This case is for removing noise from our test data.

#newer_test_data = []
#for i in range(0, len(test_Data)):
#    x = test_Data[i]
#    x = x.astype('float32')
#    img = cv2.medianBlur(x, 3)
#    newer_test_data.append(img)
#test_Data = np.array(newer_test_data)
#print(plt.imshow(newer_test_data[0]))

### medianBlur function sourced from "https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void%20medianBlur(InputArray%20src,%20OutputArray%20dst,%20int%20ksize)" ###


# ### Data Augmentation Technique
# ###### This technique was performed on trained images by creating 5 subset of each trained image before using it in our Model, but this has not made much of a difference.

# #### Function to shift the images by given dimension for better Accuracy

# In[ ]:


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image


# #### Creating Augmented Dataset after shifting images by 1 pixel to the top,bottom,left and right

# In[ ]:


#Creating Augmented Dataset

#train_data_augmented = [image for image in train_data]
#train_label_augmented = [image for image in train_label]
#for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
#   for image, label in zip(train_data,train_label):
#        train_data_augmented.append(shift_image(image, dx, dy))
#        train_label_augmented.append(label)

#train_data = np.array(train_data_augmented)
#train_label = np.array(train_label_augmented)


# In[ ]:


#print(train_data.shape)
#print()
#print(train_label.shape)

