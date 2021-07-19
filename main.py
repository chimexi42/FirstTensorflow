import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# print(tf.__version__)


# WORKING WITH MNIST TO DO NEURAL NETWORKS
(x_train, y_train), (x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(60000, 28,28,1).astype("float32")/255
x_test = x_test.reshape(10000,28,28,1).astype("float32")/255

print(x_train[0:20])

# model = keras.Sequential(
#     [
#         keras.layers.InputLayer(shape=(28*28)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(10),
#     ]
# )


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss = 'SparseCategoricalCrossentropy',
    optimizer = 'Adam',
    metrics=['accuracy']
)


model.summary()
model.fit(x_train,y_train, batch_size=32, epochs=5, verbose=2)

model.evaluate(x_test,y_test, batch_size = 32, verbose =2)


features = model.predict(x_train)

print(features.shape)




for i in range(10, 20):
    plt.imshow(x_train[i][:,:,0], cmap = plt.cm.binary)
    # plt.show()





