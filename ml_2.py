import pandas as pd

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc)

nyc.Date.values.reshape(-1, 1)

from sklearn.model_selection import train_test_split

"""
X train and y train are the actual data that we feed it. Makes up 75% of our original data
x test and y test is the other 25% of the data that we use to check our work and verify that the model is predicting correctly
Data is two dimensional, target is one dimensional
"""

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method expects the s amples and the targets for training
linear_regression.fit(X=X_train, y=y_train)


# print(linear_regression.coef_)
# print(linear_regression.intercept_)


predicted = linear_regression.predict(X_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]):  # check every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

# lambda implements y = mx + b
predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_

print(predict(2021))

# visualize the data with seaborn

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
y = predict(x)

import matplotlib.pyplot as plt

line = plt.plot(x, y)

plt.show()
