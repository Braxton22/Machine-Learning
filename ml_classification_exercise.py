# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features.
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris
# virginica. Each sample’s features are the sepal length, sepal width, petal
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower
# that protect the smaller inside petals before the flower buds bloom.

# EXERCISE
# load the iris dataset and use classification to see if the expected and predicted species match up
from sklearn import datasets
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11
)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)

expected = target_test

# display the shape of the data, target and target_names
"""
print(iris["data"].shape)
print(iris.target)
print(iris["target_names"])
"""
# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

first10predicted = []

for number in predicted[:20]:
    if number == 0:
        first10predicted.append(iris["target_names"][0])
    elif number == 1:
        first10predicted.append(iris["target_names"][1])
    else:
        first10predicted.append(iris["target_names"][2])

first10expected = []

for number in expected[:20]:
    if number == 0:
        first10predicted.append(iris["target_names"][0])
    elif number == 1:
        first10predicted.append(iris["target_names"][1])
    else:
        first10predicted.append(iris["target_names"][2])

print(first10predicted)
print(first10expected)

# display the values that the model got wrong

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)

# visualize the data using the confusion matrix

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(3), columns=range(3))

figure = plt2.figure(figsize=(3, 3))

axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.show()
