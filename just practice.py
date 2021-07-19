import numpy as np
from sklearn import preprocessing
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd


le = preprocessing.LabelEncoder()

le.fit([1,2,2,6])
print(le.classes_)

print(le.transform([1,1,2,6]))

le.fit(['paris', 'paris', 'tokyo', 'amsterdam'])

print(le.classes_)
print(le.transform(['tokyo', 'tokyo','tokyo', 'paris']))


exampleString = '''
    Jessica is 15 years old, and Daniel is 27 years old.
    Edward is 97, and his grandfather, Oscar is 102
'''

ages = re.findall(r'\d{1,3}', exampleString)
names = re.findall(r'[A-Z][a-z]*', exampleString)


print(ages)
print(names)



# Kneighbours classification

iris = datasets.load_iris()
x = iris.data
y= iris.target
# print(x)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

for i in [1,5,10,20,50,100]:
    model = KNN(n_neighbors = i, weights='uniform')
    model.fit(x_train, y_train)
    print(i, model.score(x_test,y_test))




year = (1999, 2003, 2011, 2017)
month = ('Mar', 'Jun', 'Jan', 'Dec')
day = (11,21,13,5)

print(zip(year, month,day))