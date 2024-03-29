# Logistic Regression Implementation

This repository contains implementations of Logistic Regression algorithms in Python, showcasing the core concepts of this crucial machine learning technique with a focus on binary classification tasks, such as purchase decision. The implementations are divided into two primary classes:

## Logistic Regression =  Sigmoid Classification

Logistic Regression is a powerful statistical method used for binary classification tasks. It extends the linear regression framework to model the probability that a given input belongs to a particular category.

### Logistic Function
The logistic function, or sigmoid function, is what differentiates logistic regression from linear regression. It maps any input value to a value between 0 and 1, making it suitable for estimating probabilities.

### Binary Classification
In the context of logistic regression, binary classification refers to categorizing data points into one of two groups (e.g., spam or not spam). Logistic regression calculates the probability that a data point belongs to a particular category.

### The Formula
The core of Logistic Regression is represented by the equation: 

``` 
fx = w * x + B

sigmoid(z) = 1/1 + e^-(z)

fz = sigmoid(z)
```

- \( sigma(y) \) is the predicted probability.
- \(x\) is your input.
- \(w\) (weight) and \(B\) (bias) are parameters adjusted during the training process.

### Weight and Bias
- **Weight (w):** Determines the impact of each feature on the outcome. In logistic regression, weights adjust how strongly each feature predicts the category.
- **Bias (B):** Controls the point at which the sigmoid curve transitions from predicting one class to another, essentially setting the classification threshold.

## 1. `SimpleLogisticRegression`
This class implements Logistic Regression using a `single feature` (e.g., the age in an purchase decision). It is designed to illustrate the basic principles of Logistic Regression, including fitting a model, making predictions, and understanding the underlying mathematics. Key features:

- **No dependencies on external libraries**: Pure Python implementation.
- **Fit Method**: `fit(x, y, lr)` where `x` is the feature (e.g., keyword frequency), `y` is the binary outcome (purchase or no purchase), and `lr` is the learning rate.
- **Predict Method**: `predict(x)` used for making predictions on new data points.
- **Predict Method**: `predict_proba(x)` used for making predictions as a range between 0 - 100.
- **Cost Function**: Evaluation of the model performance using a logistic loss function.
- **Gradient Descent**: Optimization of the model parameters.

## 2. `MultipleLogisticRegression`
Expands the concepts to handle `multiple features` (like age, income, etc.). 

- **Standard Implementation**: Traditional approach, ideal for deeper understanding of the mathematical operations.
- **Fit Method**: Adapted for multiple features to optimize the weights and bias.
- **Predict Method**: Extended to handle predictions with multiple inputs.
- **Cost Function and Gradient Descent**: Specifically designed for the logistic regression framework.

## Visualizations

1. **Logistic Regression Curve**:
   ![Logistic Regression Curve](logistic-regression.png)
   - *Description*: The plot shows the sigmoid curve representing the probability of an purchase on a single feature (e.g., age). The curve demonstrates how the probability changes from 0 to 1.

2. **Cost Function and Gradient Descent Visualization**:
   ![Cost Function and Gradient Descent](gradient-descent.png)
   - *Description*: This plot illustrates the 'cost bowl' curve of the logistic loss function and the progression of the gradient descent algorithm towards the minimum cost. It provides an intuitive understanding of how the model optimizes its parameters.
