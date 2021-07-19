import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import  to_categorical
from keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
# x_train = x_train.astype('float32')/255.0
#
# y_train = to_categorical(y_train)
# model = Sequential([
#     Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
#     MaxPooling2D((2,2)),
#     Flatten(),
#     Dense(100, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#
#
# optimizer = SGD(learning_rate=0.01, momentum=0.9)
# model.compile(
#     optimizer=optimizer,
#     loss = 'categorical_crossentropy',
#     metrics=['accuracy']
# )
#
#
# history = model.fit(x_train, y_train, epochs=10, batch_size=32)
#
# image = random.choice(x_test)
#
# plt.imshow(image, cmap= plt.get_cmap('gray'))
# plt.show()
#
# image = (image.reshape((1,28,28,1))).astype('float32')/255.0
#
# digit= np.argmax(model.predict(image)[0], axis = -1)
# print("Prediction: ", digit)


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# print(train_labels[4])
# print(test_labels[5])


class_names =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
              'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[7])
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# # plt.show()
#
#
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
#     ])
#
#
# model.compile(optimizer ='adam', loss = "sparse_categorical_crossentropy", metrics= ['accuracy'])
#
# model.fit(train_images, train_labels, epochs = 5)
#
#
# prediction = model.predict(test_images)
#
# print(np.argmax(prediction[0]))
# print(class_names[np.argmax(prediction[0])])
#
# print(np.argmax(prediction[4]))
# print(class_names[np.argmax(prediction[4])])
#
#
# for i in range(5):
#     plt.grid()
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel('Actual: '+ class_names[test_labels[i]])
#     plt.title('Predictions: ' + class_names[np.argmax(prediction[i])])
#     plt.show()
#
#
#
#


num_of_flats = (1,2,3,4,5,6,7)
prices_of_houses = (1000,2000,3000,4000,5000,600,7000)

model = keras.Sequential([keras.layers.Dense(units = 1, input_shape=[1])])
model.compile(optimizer ='sgd', loss = 'mean_squared_error')

model.fit(num_of_flats, prices_of_houses, epochs = 500)


prediction = model.predict([15])
prediction = model.predict([20])

print('Price of 10 flats is {}'.format(prediction))


model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, input_shape=[1]))
model.compile(optimizer = 'sgd', loss ='mean_squared_error')

xs = np.array([[1],[2],[3],[4]])
ys = np.array([[1],[3],[5],[7]])
model.fit(xs, ys, epochs = 1000)

print(model.predict(np.array([[4]])))

print(model.predict(np.array([[5]])))



















