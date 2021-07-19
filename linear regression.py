try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist
except:
    print('Library not found')

x = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])

y = np.array([4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0])

model= tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='sgd', loss=tf.keras.losses.mean_absolute_error)
model.fit(x,y, epochs = 100)
prediction = model.predict(x=[10.0, 20.0, 30.0, 40.0])

print('The predicted Value is {}'.format(prediction))