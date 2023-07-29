#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


# In[5]:


dataFile = "iris.csv"
print(dataFile)


# In[6]:


df = pd.read_csv(dataFile)
df.head()


# In[7]:


X = df.iloc[:,0:4].values
y = df.iloc[:,4].values


# In[8]:


print(X[0:5])
print(y[0:5])


# In[9]:


print(X.shape)
print(y.shape)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)


# In[11]:


print(y1)


# In[12]:


Y = pd.get_dummies(y1).values
print(Y[0:5])


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[14]:


print(X_train[0:5])


# In[15]:


print(y_train[0:5])


# In[16]:


print(X_test[0:5])


# In[17]:


print(y_test[0:5])


# In[24]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
  ])
model


# In[25]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[26]:


with tf.device('/gpu:0'):
	model.fit(X_train, y_train, batch_size=50, epochs=100)


# In[27]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# In[28]:


y_pred = model.predict(X_test)
y_pred


# In[29]:


actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")


# Teniendo una modificación pequeña, solamente modificando solamente las funciones de activación de 'sigmoid' a 'relu' de las primeras 2 capas de 10 neuronas cada una y la utilización de la GPU para el entrenamiento del modelo, se obtuvo ya un accuracy con los datos de testing de 100%, por esto, no se modifico algo adicional.
