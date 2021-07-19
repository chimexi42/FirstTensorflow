import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print(x_train.shape)
print(x_test.shape)

dataframe = pd.read_csv('ecg.csv', header=None)
raw_data = dataframe.values
print(dataframe.head())

labels = raw_data[:, -1]
print(labels)

data = raw_data[:, 0:-1]
print(data)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=2, test_size=0.2)

# Normalize the data

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

normal_train_data = (train_data - min_val)/ (max_val - min_val)
normal_test_data = (test_data - min_val)/ (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

#
# plt.grid()
# plt.plot(np.arange(140), normal_train_data[0])
# plt.title('A Normal ECG')
# plt.show()
#
# plt.grid()
# plt.plot(np.arange(140), anomalous_train_data[0])
# plt.title('Anomalous ECG')
# plt.show()

# Build  the model


class anomalyDetector(Model):
    def __init__(self):
        super(anomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(14, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(140, activation='sigmoid')
        ])

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = anomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(normal_train_data, normal_train_data,
                          epochs=20,
                          batch_size=512,
                          validation_data=(test_data, test_data),
                          shuffle=True)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label= 'validation Loss')
plt.legend()
plt.show()


encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_imgs[0], 'r')
plt.fill_between(np.arange((140), decoded_imgs[0], normal_test_data[0]), color='blue')
plt.legend(labels = ['input', 'Recnstruction', 'Error'])
plt.show()
