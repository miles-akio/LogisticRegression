# Logistic Regression Project

In this project, I implemented logistic regression and applied it to two real-world datasets. I explored both standard logistic regression and regularized logistic regression to understand how the model performs with linear and non-linear decision boundaries.

---

## Table of Contents

* [1. Packages](#1-packages)
* [2. Logistic Regression](#2-logistic-regression)

  * [2.1 Problem Statement](#21-problem-statement)
  * [2.2 Loading and Visualizing the Data](#22-loading-and-visualizing-the-data)
  * [2.3 Sigmoid Function](#23-sigmoid-function)
  * [2.4 Cost Function](#24-cost-function-for-logistic-regression)
  * [2.5 Gradient Computation](#25-gradient-for-logistic-regression)
  * [2.6 Learning Parameters with Gradient Descent](#26-learning-parameters-using-gradient-descent)
  * [2.7 Plotting the Decision Boundary](#27-plotting-the-decision-boundary)
  * [2.8 Evaluating the Model](#28-evaluating-logistic-regression)
* [3. Regularized Logistic Regression](#3-regularized-logistic-regression)

  * [3.1 Problem Statement](#31-problem-statement)
  * [3.2 Loading and Visualizing the Data](#32-loading-and-visualizing-the-data)
  * [3.3 Feature Mapping](#33-feature-mapping)
  * [3.4 Regularized Cost Function](#34-cost-function-for-regularized-logistic-regression)
  * [3.5 Regularized Gradient Computation](#35-gradient-for-regularized-logistic-regression)
  * [3.6 Learning Parameters with Gradient Descent](#36-learning-parameters-using-gradient-descent)
  * [3.7 Plotting the Decision Boundary](#37-plotting-the-decision-boundary)
  * [3.8 Evaluating the Model](#38-evaluating-regularized-logistic-regression-model)

---

## 1. Packages

To implement logistic regression, I used the following Python packages:

* **numpy** – for numerical computations and array operations
* **matplotlib** – for data visualization
* **utils.py** – contains helper functions such as plotting and feature mapping

```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
%matplotlib inline
```

---

## 2. Logistic Regression

In this part of the project, I implemented logistic regression to predict whether a student gets admitted into a university based on two exam scores.

### 2.1 Problem Statement

I had historical data containing scores of previous applicants and their admission results. The goal was to estimate the probability of admission for new applicants based on their exam scores using a logistic regression classifier.

---

### 2.2 Loading and Visualizing the Data

I loaded the dataset using a helper function, which gave me:

* `X_train` – the exam scores
* `y_train` – the admission outcome (1 = admitted, 0 = not admitted)

I explored the data by printing values and shapes and then visualized it on a 2D plot to understand the distribution.

```python
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc="upper right")
plt.show()
```

---

### 2.3 Sigmoid Function

The sigmoid function is used to map any real-valued number into the range \[0, 1], which is interpreted as a probability in logistic regression.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

### 2.4 Cost Function for Logistic Regression

I implemented the logistic regression cost function:

$$
J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \Big[-y^{(i)}\log(f_{\mathbf{w},b}(x^{(i)})) - (1-y^{(i)})\log(1-f_{\mathbf{w},b}(x^{(i)})) \Big]
$$

This measures how well the predicted probabilities match the true labels.

---

### 2.5 Gradient for Logistic Regression

I calculated the gradients of the cost function with respect to the weights and bias:

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (f(x^{(i)}) - y^{(i)})
$$

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (f(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

These gradients were used to update the parameters during gradient descent.

---

### 2.6 Learning Parameters Using Gradient Descent

I applied batch gradient descent to learn the optimal weights and bias. I verified that the cost consistently decreased over iterations.

---

### 2.7 Plotting the Decision Boundary

Using the learned parameters, I plotted the decision boundary that separates admitted and not admitted students.

```python
plot_decision_boundary(w, b, X_train, y_train)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc="upper right")
plt.show()
```

---

### 2.8 Evaluating Logistic Regression

I implemented a `predict` function to make predictions for new data points and calculated the training accuracy, which was approximately 92%.

---

## 3. Regularized Logistic Regression

In this part, I implemented regularized logistic regression to predict whether microchips from a factory pass quality assurance.

### 3.1 Problem Statement

The dataset contained two test scores per microchip and the QA outcome (1 = accepted, 0 = rejected). The goal was to predict whether new microchips should be accepted or rejected.

---

### 3.2 Loading and Visualizing the Data

I loaded the microchip dataset, explored it, and visualized it. Unlike the previous dataset, a straight-line decision boundary could not separate the classes.

```python
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(loc="upper right")
plt.show()
```

---

### 3.3 Feature Mapping

To handle the non-linear nature of the data, I mapped the features into polynomial terms up to the sixth power, transforming each example into a 27-dimensional vector.

```python
mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
```

---

### 3.4 Cost Function for Regularized Logistic Regression

I modified the cost function to include a regularization term:

$$
J(\mathbf{w},b) = \text{Cost without regularization} + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2
$$

This helps prevent overfitting when using higher-dimensional feature vectors.

---

### 3.5 Regularized Gradient Computation

I computed the gradients including the regularization term:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (f(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j
$$

The bias term `b` was not regularized.

---

### 3.6 Learning Parameters Using Gradient Descent

I ran gradient descent using the regularized cost and gradients. The parameters converged to values that minimized the regularized cost.

---

### 3.7 Plotting the Decision Boundary

With the learned parameters, I plotted a nonlinear decision boundary that successfully separated the accepted and rejected microchips.

---

### 3.8 Evaluating the Model

Finally, I evaluated the regularized logistic regression model using the training dataset and observed improved generalization compared to standard logistic regression due to the regularization.

---

This project helped me understand the full implementation of logistic regression, the role of the sigmoid function, gradient computation, and how regularization can prevent overfitting in high-dimensional feature spaces.

---
