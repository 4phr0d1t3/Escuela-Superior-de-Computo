import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow import keras


# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='relu'))

# opt=keras.optimizers.SGD(learning_rate=0.03)
# opt=keras.optimizers.Adam(learning_rate=0.15)

model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['binary_accuracy'])

with tf.device('/gpu:0'):
    model.fit(training_data, target_data, epochs=50, verbose=2)

print(model.predict(training_data).round())