# Logistic Regression with L1 & L2 Regularization

## Parameters
* learning_rate = 0.01
* n_epoch = 100
* penalty = [None, 'l1', 'l2']
* C = 0.01 (1/lambda)
* tolerance = 0.0001
* pred_threshold = 0.5

## Attributes
* costs_: cross-entropy loss values
* thetas_: weights or coefficients
* n_epoch_reached_: number of epoch iterations completed during training
* pred_proba_: predicted probabilities of success per instance
* pred_: predicted class per instance
* conf_mat_: confusion matrix

