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

## Methods
* fit(self, X_train, y_train)
* predict(self, X_test)
* predict_proba(self, X_test)
* confusion_matrix(self, y_test, y_pred)
* model_evel(self, y_test, y_pred, y_pred_proba)

## Functions
* standardize_features(df)
* auc_roc_curve(y_actual, y_pred_proba)
* minimizing_cost_func_curve(costs, n_epoch_reached)
