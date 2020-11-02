import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import perceptron as pt
import adaline as ad
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


# pg 91
# It often requires some experimentation to find a good learning for optimal
# convergence. Here we'll experiment with a learning rate of '0.1' and '0.0001',
# and plot the cost functions versus the epochs to see how well the Adaline
# implementation learns from the training data.

# FYI, the learning rate and number of epochs are the hyperparameters (tuning parameters)
# of the perceptron and Adaline learning algorithms.

# Findings:
#   A big learning rate may over shoot the global minimum
#   A small learning rate value moves gradually to the direction of the global minimum, and
#   therefore requires more epochs (iterations) to reach it.

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = ad.AdalineGD(n_iter=300, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = ad.AdalineGD(n_iter=300, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
