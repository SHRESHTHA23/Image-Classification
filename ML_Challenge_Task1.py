#!/usr/bin/env python
# coding: utf-8

# # Task 1

# ### Importing Required Packages

# In[ ]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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


# In[ ]:


plt.figure(figsize=(6,6))
for i in range (1,7):
    plt.subplot(2,3,i)
    plt.imshow(feature[i].reshape(28,28))


# ### Splitting our Dataset into Train, Validation and Test Set using Stratification

# In[ ]:


train_data, val_data, train_label, val_label = train_test_split(feature,label,stratify = label,test_size = 0.3,random_state = 640)


# In[ ]:


val_data, test_data, val_label, test_label = train_test_split(val_data,val_label,stratify = val_label,test_size = 0.5,random_state = 640)


# In[ ]:


train_data.shape


# In[ ]:


train_label.shape


# In[ ]:


#Validating the stratification

unique,counts = np.unique(train_label,return_counts = True)
print(dict(zip(unique,counts)))


# In[ ]:


val_data.shape


# In[ ]:


val_label.shape


# In[ ]:


#Validating the stratification

unique,counts = np.unique(val_label,return_counts = True)
print(dict(zip(unique,counts)))


# In[ ]:


test_data.shape


# In[ ]:


test_label.shape


# In[ ]:


#Validating the stratification

unique,counts = np.unique(test_label,return_counts = True)
print(dict(zip(unique,counts)))


# ### CNN Classifier

# In[ ]:


final_train_label = to_categorical(np.array(train_label), num_classes = 27)
final_val_label = to_categorical(np.array(val_label), num_classes = 27)


# In[ ]:


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

# Model Structure sourced from ML Notebook


# In[ ]:


# This is to prevent overfitting

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


CNN_Metrics[['loss', 'val_loss']].plot()
plt.show()


# In[ ]:


CNN_Metrics[['accuracy', 'val_accuracy']].plot()
plt.show()


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


# #### Predicted Value and Corresponding Letter (in Test Data)

# In[ ]:


y_pred[38]


# In[ ]:


x = test_data[38]
plt.imshow(x.reshape(28,28))
plt.show()
print(test_label[40])

