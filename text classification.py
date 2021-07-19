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


# data = keras.datasets.imdb
# (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)
#
# print(train_data[0])
#
#
# word_index =data.get_word_index()
#
# word_index= {k:(v+3) for k,v in word_index.items()}
#
# word_index["<PAD>"] =0
# word_index["<START>"] =1
# word_index["<UNK>"] =2
# word_index["<UNUSED>"] =3
#
# reverse_word_index = dict([(value, key) for (key, value) in word_index.item()])


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test= tf.keras.utils.normalize(x_test, axis =1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

model.fit(x_train, y_train, epochs = 3)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)

model.save('number_reader.model')

new_model = tf.keras.models.load_model('number_reader.model')
predictions = new_model.predict([x_test])
print(predictions)

print(np.argmax(predictions[0]))



plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()