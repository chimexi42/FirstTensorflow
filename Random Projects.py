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


mushroom_df = pd.read_csv('mushrooms.csv')

print(mushroom_df.head())

print(mushroom_df.shape)

print(mushroom_df.describe)

print(mushroom_df.groupby(['class', 'odor']).count())

labels = mushroom_df['class']
features = mushroom_df.drop(columns = ['class'])

print(labels[0:5])
print(features[0:5])


labels.replace('p', 0, inplace= True)
labels.replace('e', 1, inplace= True)


features = pd.get_dummies(features)

print(features[0:5])

features = features.values.astype('float32')
labels = labels.values.astype('float32')


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.2)

features_train, features_validation, labels_train, labels_validation= train_test_split(features_train, labels_train, test_size=.2)


model = keras.Sequential([keras.layers.Dense(32, input_shape=(117,)),
                          keras.layers.Dense(20, activation =tf.nn.relu),
                          keras.layers.Dense(2, activation='softmax')


                          ])

model.compile(optimizer ='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['acc'])


history = model.fit(features_train,labels_train, epochs = 20, validation_data=(features_validation, labels_validation))

prediction_features = model.predict(features_test)
performance = model.evaluate(features_test, labels_test)

print('\nPerformance: {}'.format(performance))
# print('Predictions: {}'.format(prediction_features))


history_dict = history.history
print(history_dict.keys())
print('History Dict: {}'.format(history_dict))

# checking over fitting
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']


epochs = range(1, len(acc) +1)
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, '*', label ="Validation loss")
plt.title("Training and validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()