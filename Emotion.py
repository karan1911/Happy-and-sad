#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import tensorflow as tf
import keras 

from keras.models  import Sequential
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense , Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


# In[2]:


path1 = 'E:\Emotin'
face = ['Happy' , 'Sad']


# In[3]:


for i in face:
    folders = os.path.join(path1 , i)
    print(folders)


# In[4]:


for i in face:
    folders = os.path.join(path1 , i)
    for image in os.listdir(folders):
        image_path = os.path.join(folders , image)
        print(image_path)


# In[5]:


for i in face:
    folders = os.path.join(path1 , i)
    for image in os.listdir(folders):
        image_path = os.path.join(folders , image)
        image_array = cv2.imread(image_path)
        plt.imshow(image_array)
        break
# just to show image


# In[6]:


image_size = 200
path1 = 'E:\Emotin'
face = ['Happy' , 'Sad']

input_image = []
for i in face:
    folders = os.path.join(path1 , i)
    label = face.index(i)
    for image in os.listdir(folders):
        image_path = os.path.join(folders , image)
        image_array = cv2.imread(image_path)
        image_array = cv2.resize(image_array , (image_size , image_size))
        input_image.append([image_array , label])


# In[7]:


len(input_image)


# In[8]:


np.random.shuffle(input_image)


# In[9]:


X = []
Y = []

for X_values , labels in input_image:
    X.append(X_values)
    Y.append(labels)


# In[10]:


X = np.array(X)
Y = np.array(Y)


# In[11]:


X_train = X[0 : 3200]
Y_train = Y[0 : 3200]


X_test = X[3200 : 4000]
Y_test = Y[3200:  4000]


# In[12]:


len(Y_test)


# In[13]:


X_train = X_train/ 255
X_test  = X_test / 255


# In[14]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())
model.add(Dense(128 , activation='relu' , input_shape = X.shape[1:]))# Dense layer
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))


# In[15]:


adam = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer='adam' , loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[16]:


model.fit(X_train,Y_train , epochs=20,validation_split=0.1  )


# In[17]:


pred = model.predict(X_test)


# In[18]:


pred_classes = pred.argmax(axis = 1)


# In[19]:


from sklearn.metrics import confusion_matrix ,accuracy_score, recall_score , precision_score ,f1_score,classification_report


# In[20]:


confusion_matrix(Y_test,pred_classes)


# In[21]:


print(classification_report(Y_test,pred_classes))


# In[ ]:


6000/54


# In[ ]:


# TEXT ---> CON , POOling
# flttern your Y


# In[ ]:


# CCN
# VGG16 , VV19 , RESnet


# In[ ]:


# cnn ----> COnv , POOLING
# FLTTEN 


# In[ ]:


model.save("Happy_Sad.h5")


# In[ ]:


from keras.models import load_model


# In[ ]:


load = load_model("Happy_Sad.h5")


# In[ ]:


load.predict(X_test)


# In[ ]:


confusion_matrix(Y_test,pred_classes)


# In[ ]:


pwd


# In[ ]:





# In[22]:


import cv2
import numpy as np
from keras.models import load_model

model=load_model(r"C:\\Users\\Dell\\Neural_Networks\Happy_Sad.h5")

results={0:'happy',1:'sad'}
GR_dict={0:(0,255,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0)

haarcascade = cv2.CascadeClassifier(r"C:\Users\Dell\Downloads\haarcascade_frontalface_default.xml")
while True:
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1) 

    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(200,200))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,200,200,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

    cv2.imshow('Liv Camera',   im)
    key = cv2.waitKey(10)
    
    if key == 27: # use the escape key
        break

cap.release()

cv2.destroyAllWindows()


# In[ ]:




