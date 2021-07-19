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
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)


print('Xtrain:', x_train)
print('Xtest:', x_test)
print('Ytrain:', y_train)


# Flatten the data

x_train = x_train.reshape(-1, 28*28).astype(np.float32)/255
x_test = x_test.reshape(-1, 28*28).astype(np.float32)/255
print(x_train.ndim)
print(x_train.shape)

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)
acc = confusion_matrix(y_test, prediction)
print(acc)

def accuracy(confusuion_matrix):
    diagonal = confusuion_matrix.trace()
    elements = confusuion_matrix.sum()
    return diagonal/elements

print(accuracy(acc))





# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(10))
#
#
# model.compile(
#     loss = keras.losses.SparseCategoricalCrossentropy(from_logits= True),
#     optimizer = keras.optimizers.Adam(learning_rate = 0.001),
#     metrics= ['accuracy']
# )
#
# model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
#
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)
#
#
# feature = model.predict(x_train)
# print(feature.shape)
