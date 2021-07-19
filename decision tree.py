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

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]

header = ['color', 'diameter', 'label']


def unique_vals(rows, col):
    '''find the unique values for a column in the datset'''
    return set([row[col] for row in rows])


unique = unique_vals(training_data, 0)
print(unique)


def class_counts(rows):
    '''counts the number of each type of ecxample in the dataset'''
