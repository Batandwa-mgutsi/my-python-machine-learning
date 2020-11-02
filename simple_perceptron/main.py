import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import perceptron as pt
import helpers as helpers

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print("URL:", s)

# Load data to Data Frame
df = pd.read_csv(s, header=None, encoding='utf-8')

# Check if we have the right data
# print(df.tail())

# Select 50 Iris-setosa and 50 Iris-versicolor flowers and transform them into
# a vector, y, with elements 1 (for versicolor) and -1 (setosa),
# We will only focus on 2 features from the 100 training examples for now:
# sepal length (1st column) and petal length (3rd column). We will assign these
# features to a feature matrix (2 columns and 100 rows), which we can visualise
# with a scatter plot.

# Select setosa and versicolor (1st 100 entries). The 4 means 'take only the
# 4th (last) feature, which is the name.
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract the 1st 100 sepal length and petal length features (i.e feature 1 and 3)
X = df.iloc[0:100, [1, 2]].values

# Plot the data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
            marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Train our Perceptron and plot the number of misclassifications(errors) for each
# epoch (iteration). The goal is that the more we train the perceptronthe less
# the misclassifications.
#
# The model is ready when the number of errors converges to and finally reaches 0.
ppn = pt.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# Test the model with the same training examples it used
helpers.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
