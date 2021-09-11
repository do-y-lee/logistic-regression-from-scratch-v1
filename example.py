import pandas as pd
from linear_model import LogisticRegression, standardize_features


train = pd.read_csv('datasets/titanic/transformed-train.csv')
test = pd.read_csv('datasets/titanic/transformed-test.csv')

X_train = train[['pclass', 'sex', 'age']]
X_train = standardize_features(X_train)
y_train = train['survived']

logit = LogisticRegression(learning_rate=0.01)
logit.fit(X_train, y_train)
logit.predict(X_train)
logit.predict_proba(X_train)

print(logit.confusion_matrix(y_train, logit.pred_))
print(logit.model_eval(y_train, logit.pred_, logit.pred_proba_))
print(logit.n_epoch_reached_)
print(logit.thetas_)
print(logit.costs_)
