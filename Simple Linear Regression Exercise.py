import pandas as pd

temperatures = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    temperatures.Date.values.reshape(-1, 1), temperatures.Value.values, random_state=11
)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X=X_train, y=y_train)

predicted = linear_regression.predict(X_test)
expected = y_test

"""
this is code for evaluating prediction accuracy
for p, e in zip(predicted[::5], expected[::5]):  # check every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")
"""

# lambda implements y = mx + b
predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_

# visualize the data with seaborn

import seaborn as sns

axes = sns.scatterplot(
    data=temperatures,
    x="Date",
    y="Value",
    hue="Value",
    palette="winter",
    legend=False,
)

axes.set_ylim(47.5, 60)

import numpy as np

x = np.array([min(temperatures.Date.values), max(temperatures.Date.values)])
y = predict(x)

import matplotlib.pyplot as plt

line = plt.plot(x, y)

plt.show()

# To answer the question at the end of the assignment:
# The temperature trend in this dataset is more positive compared to the high temperatures in January from the other dataset
# In the other dataset, the regression line is more flat whereas the regression line in this dataset is sloped upward