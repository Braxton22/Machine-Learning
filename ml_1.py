from sklearn.datasets import load_digits

digits = load_digits()

# print(digits.DESCR)  # contains the dataset's description

# print(digits.data[13])  # numpy array that contains the 1797 samples

# print(digits.data.shape)

# print(digits.target[13])  #

# print(digits.target.shape)  #

# print(digits.images[13])


"""
import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    # displays multichannel (RGB) or single channel (grayscale)
    # image data
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # rmove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(target)  # target value of the image
plt.tight_layout()
plt.show()
"""
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)

print(data_train.shape)
print(target_train.shape)
print(data_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)

print(data_test)

expected = target_test

print(predicted[:20])
print(expected[:20])


# zip just iterates through two+ objects at the same time
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7, 6))

axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.show()
