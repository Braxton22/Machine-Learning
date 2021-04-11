""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn import datasets
import pandas as pd

diabetes = sklearn.datasets.load_diabetes()

Diabetes = pd.DataFrame(
    data=np.c_[diabetes["data"], diabetes["target"]],
    columns=diabetes["feature_names"] + ["target"],
)

# how many sameples and How many features?
print(diabetes.data.shape)
"""As you can see, there are 10 features and 442 samples"""

# What does feature s6 represent?
print(diabetes.DESCR)
"""it represents the glucose in blood sugar level"""

# print out the coefficient

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method expects the s amples and the targets for training
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)

# print out the intercept

print(linear_regression.intercept_)

# create a scatterplot with regression line

predicted = linear_regression.predict(X_test)
expected = y_test

import matplotlib.pyplot as plt

scatter = plt.plot(predicted, expected, ".")

x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()
