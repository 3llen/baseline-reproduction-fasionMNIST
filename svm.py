from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import gzip
import os
import matplotlib.pyplot as plt
from datetime import datetime

from utils import mnist_reader
# using external library to load fashion mnist data

start_time = datetime.now()

X_train, y_train = mnist_reader.load_mnist(path='data', kind='train')
X_test, y_test = mnist_reader.load_mnist(path='data', kind='t10k')
print("done loading the data")

# fitting the data with preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
# full working svc model with smaller subset with gives 0.8636 of accu
small_X_train = X_train[0:10000]
small_y_train = y_train[0:10000]
small_X_test = X_test[0:10000]
small_y_test = y_test[0:10000]

classifier = SVC(C=10.0, kernel='rbf', verbose=True)
classifier.fit(small_X_train, small_y_train)
predicted = classifier.predict(small_X_test)
time_used = (datetime.now() - start_time).total_seconds()
acc_score = accuracy_score(small_y_test, predicted)
print("acc_score:", acc_score, "Time Used:", time_used, "seconds")
'''

classifier = SVC(C=10.0, kernel='rbf', verbose=True,)
classifier.fit(X_train, y_train)
print('done training the model')

predicted = classifier.predict(X_test)
print('done predicting on test set')

time_used = (datetime.now() - start_time).total_seconds()
acc_score = accuracy_score(y_test, predicted)
print("acc_score:", acc_score, "Time Used:", time_used + "seconds")
