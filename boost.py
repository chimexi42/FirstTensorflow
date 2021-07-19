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
from sklearn import datasets
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from keras.losses import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn. tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier


#
# iris = datasets.load_iris()
#
# x = iris.data
# y = iris.target
#
#
#
# print(x[0:7, :])
#
# print('\n')
#
# print(y)
#
# num_samples, num_feature = iris.data.shape
# print(num_samples)
# print(num_feature)
#
# print(iris.target_names)
#
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
#
# train = xgb.DMatrix(x_train, label =y_train)
# test = xgb.DMatrix(x_test, label=y_test)
#
# param = {
#     'max_depth' : 4,
#     'eta':0.3,
#     'objective': 'multi: softmax',
#     'num_class': 3
# }
# epochs = 10
#
# model = xgb.train(param, train, epochs)
#
# predictions = model.predict(test)
# print(predictions)

# dataset = load_boston()
# x_train, x_test, y_train, y_test = train_test_split(
#     pd.DataFrame(dataset.data, columns=dataset.feature_names),
#
#     pd.Series(dataset.target)
# )
#
# x =pd.DataFrame(dataset.data, columns=dataset.feature_names)
#
# y = pd.Series(dataset.target)
#
#
# print(x.head())
# print(y.head())
#
# regressor = xgb.XGBRegressor(
#     n_estimators=100,
#     reg_lambda=1,
#     gamma=0,
#     max_depth=3
# )
#
# regressor.fit(x_train, y_train)
# y_pred = regressor.predict(x_test)
#
# print(y_pred)
#
# print('Predicted: \n {} '.format(y_pred))
#
# performance = mean_squared_error(y_test,y_pred)
# print(performance)


dataset = pd.read_csv('mushrooms.csv')
dataset = dataset.sample(frac=1)

dataset.columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                   'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                   'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                   'stalk-surface-below-ring', 'stalk-color-above-ring',
                   'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                   'ring-type', 'spore-print-color', 'population', 'habitat']

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])
    print(label)


print(dataset.info())

x = dataset.drop(['target'], axis =1)
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = DecisionTreeClassifier(criterion='entropy', max_depth= 1)
AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=400, learning_rate=1)
boostModel = AdaBoost.fit(x_train, y_train)

y_pred = boostModel.predict(x_test)

print(y_pred[0:10])
predictions = metrics.accuracy_score(y_test, y_pred)

print('The Accuracy is: {}%'.format(predictions*100))
