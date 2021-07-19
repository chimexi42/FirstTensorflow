import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import keras
from keras.layers import Conv2D


print('Tensorflow:', tf.__version__)
print('Python :', sys.version)
print('Numpy :', np.__version__)
print('matplotlib: ', matplotlib.__version__)

#
# # weights1 = [0.2,0.8,-0.5, 1.0]
# # weights2 = [0.5, -0.91, 0.26, -0.5]
# # weights3 = [-0.26,-0.27, 0.17, 0.87]
# # bias1 = 2
# # bias2 = 3
# # bias3 = 0.5
#
#
# # output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2]+ inputs[3] * weights1[3] + bias1,
#           # inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2]+ inputs[3] * weights2[3] + bias2,
#           # inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2]+ inputs[3] * weights3[3] + bias3]
#           # print(output)
#
#
# inputs = [1,2,3,2.5]
#
#
# weights = [[0.2,0.8,-0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26,-0.27, 0.17, 0.87]]
#
# biases = [2,3,0.5]
#
#
# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
#
# print(layer_outputs)
#
# inputs = [1,2,3,2.5]
# weights = [0.2, 0.8,-0.5,1.0]
# bias = 2
#
# output = np.dot(inputs, weights) + bias
# print(output)
#
#
# inputs = [1,2,3,2.5]
#
# weights = [[0.2,0.8,-0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26,-0.27, 0.17, 0.87]]
#
# biases = [2,3,0.5]
#
#
# output = np.dot(weights, inputs) + biases
# print(output)


model = keras.models.Sequential()
model.add(Conv2D(40, kernel_size=(3,3), input_shape=(64,64,3)))
model.summary()


def getPercentages(numbers):
    total = sum(numbers)
    pcts = []
    for v in numbers:
        pcts.append(v/total * 100)
        return pcts

def getGenerator(nums):
    for i in nums:
        yield  i
    
print(getPercentages(getGenerator([1,6,3])))