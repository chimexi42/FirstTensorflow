import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import math

x = np.random.randint(low =1, high = 9, size = (100,2))
y = np.prod(x, axis=1).reshape((100,1))

x = np.log(x)


# print(x)
# print(y)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=(2,)),
    tf.keras.layers.Dense(units=4, activation="relu"),
    tf.keras.layers.Dense(units=1)
])

model.compile(
    loss = 'mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.01)
)


model.fit(x,y, epochs = 250, verbose=False)
x_test = np.random.randint(low=1, high=9, size=(10,2))
y_test = np.prod(x_test, axis =1).reshape(10,1)


print(x_test)
print('\n')
print('y test', y_test)


predictions = model.predict(np.log(x_test))


for i, prediction in enumerate(predictions):
    print(f"{x_test[i][0]} * {x_test[i][1]} = {math.floor(prediction)}")


lin = [4,2]
print(np.product(lin))