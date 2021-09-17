# Logistic Regression with L1 & L2 Regularization

## Overview

> Logistic regression uses linearly combined features and the sigmoid function to convert 
> log-odd values from the linear form into probabilities. In this example, the cross-entropy loss 
> (negative log-likelihood) is used as the cost function to calculate the theta parameters 
> (weights/coefficients) by minimizing the function through gradient descent. L1 (Lasso) 
> and L2 (Ridge) regularization methods are used to shrink the thetas in order to 
> prevent overfitting.


## Concepts

* Linear model and sigmoid function
* Cross-entropy loss (cost) function
* L1 and L2 regularization
* Gradients - partial derivatives per theta parameter
* Gradient descent


## Package Details

### Parameters
* learning_rate = 0.01
* n_epoch = 100
* penalty = [None, 'l1', 'l2']
* C = 0.01 (1/lambda)
* tolerance = 0.0001
* pred_threshold = 0.5

### Attributes
* costs_: cross-entropy loss values
* thetas_: weights or coefficients
* n_epoch_reached_: number of epoch iterations completed during training
* pred_proba_: predicted probabilities of success per instance
* pred_: predicted class per instance
* conf_mat_: confusion matrix

### Methods
* fit(self, X_train, y_train)
* predict(self, X_test)
* predict_proba(self, X_test)
* confusion_matrix(self, y_test, y_pred)
* model_eval(self, y_test, y_pred, y_pred_proba)

### Functions
* standardize_features(df)
* auc_roc_curve(y_actual, y_pred_proba)
* minimizing_cost_func_curve(costs, n_epoch_reached)


## Code Example

```python
import pandas as pd
from linear_model import LogisticRegression, standardize_features, auc_roc_curve, minimizing_cost_func_curve


train = pd.read_csv('datasets/titanic/transformed-train.csv')
test = pd.read_csv('datasets/titanic/transformed-test.csv')

X_train = train[['pclass', 'sex', 'age']]
X_train = standardize_features(X_train)
y_train = train['survived']

logit = LogisticRegression(penalty='l2')
logit.fit(X_train, y_train)
logit.predict(X_train)
logit.predict_proba(X_train)

print(logit.confusion_matrix(y_train, logit.pred_))
print(logit.model_eval(y_train, logit.pred_, logit.pred_proba_))
print(logit.n_epoch_reached_)
print(logit.thetas_)
print(logit.costs_)

# plot roc curve
auc_roc_curve(y_train, logit.pred_proba_)
# cost func curve
minimizing_cost_func_curve(logit.costs_, logit.n_epoch_reached_)
```

## Future Additions
* Learning rate - hyperparameter tuning 
* Mini-batch gradient descent
* Full unit testing code


## References
* https://explained.ai/regularization/L1vsL2.html
* https://en.wikipedia.org/wiki/Regularization_(mathematics)
* https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
* https://web.stanford.edu/~jurafsky/slp3/5.pdf
