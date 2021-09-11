import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class LogisticRegression:
    def __init__(self, learning_rate=0.005, n_epoch=100, penalty=None, C=0.01, tolerance=1e-4, pred_threshold=0.5):
        # parameters
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.penalty = penalty
        self.C = C  # 1/lambda; inverse lambda regularization parameter
        self.tolerance = tolerance
        self.pred_threshold = pred_threshold

        # attributes
        self.costs_ = []
        self.thetas_ = []
        self.n_epoch_reached_ = 0
        self.pred_proba_ = None
        self.pred_ = None
        self.conf_mat_ = dict()

    @staticmethod
    def _logistic_sigmoid(z):
        predict_proba = 1 / (1 + np.exp(-z))
        predict_proba = np.array(predict_proba)
        return predict_proba

    @classmethod
    def _cross_entropy_loss(cls, y_train, y_pred_proba):
        return np.sum(-1 * (y_train * np.log(y_pred_proba) + (1 - y_train) * np.log(1 - y_pred_proba)))

    @classmethod
    def _cross_entropy_loss_l1(cls, y_train, y_pred_proba, thetas, C):
        """ L1 regularization with inverse of lambda represented with C (regularization parameter).
            This approach follows sklearn's regularized cost function for logistic regression.

        References
        ----------
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        """
        l1_regularization = np.sum(np.absolute(thetas))
        return C * np.sum(-1 * (y_train*np.log(y_pred_proba) + (1-y_train)*np.log(1-y_pred_proba))) + l1_regularization

    @classmethod
    def _cross_entropy_loss_l2(cls, y_train, y_pred_proba, thetas, C):
        """ L2 regularization with inverse of lambda represented with C (regularization parameter).
            This approach follows sklearn's regularized cost function for logistic regression.

        References
        ----------
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        """
        l2_regularization = 0.5 * thetas.T.dot(thetas)
        return C * np.sum(-1 * (y_train*np.log(y_pred_proba) + (1-y_train)*np.log(1-y_pred_proba))) + l2_regularization

    def fit(self, X_train, y_train):
        n_rows = X_train.shape[0]
        n_features = X_train.shape[1]

        thetas = np.zeros(n_features + 1)
        tol = self.tolerance * np.ones(n_features + 1)
        X_train = np.c_[np.ones(n_rows), np.array(X_train)]

        for _ in range(self.n_epoch):
            z = np.dot(X_train, thetas)
            y_pred_proba = self._logistic_sigmoid(z)
            diffs = y_pred_proba - y_train

            if self.penalty == 'l2':
                gradient_update_terms = self.learning_rate * (self.C * np.dot(diffs, X_train) + np.sum(thetas))
            elif self.penalty == 'l1':
                gradient_update_terms = self.learning_rate * (self.C * np.dot(diffs, X_train) + np.sign(thetas))
            elif self.penalty is None:
                gradient_update_terms = self.learning_rate * np.dot(diffs, X_train)

            if np.all(abs(gradient_update_terms) >= tol):
                # calculate and append cost per epoch
                if self.penalty == 'l2':
                    self.costs_.append(self._cross_entropy_loss_l2(y_train, y_pred_proba, thetas, self.C))
                elif self.penalty == 'l1':
                    self.costs_.append(self._cross_entropy_loss_l1(y_train, y_pred_proba, thetas, self.C))
                elif self.penalty is None:
                    self.costs_.append(self._cross_entropy_loss(y_train, y_pred_proba))

                # gradient descent updating step if gradient_update_terms (learning_rate * gradients) >= tol
                thetas = thetas - gradient_update_terms
                # n_epoch counter
                self.n_epoch_reached_ += 1
            else:
                break

        # append optimized thetas onto self.thetas_ parameter
        for theta in thetas:
            self.thetas_.append(theta)

        return self

    def predict(self, X_test):
        z = self.thetas_[0] + np.dot(X_test, self.thetas_[1:])
        self.pred_proba_ = self._logistic_sigmoid(z)
        self.pred_ = np.where(self.pred_proba_ > self.pred_threshold, 1, 0)
        return self.pred_

    def predict_proba(self, X_test):
        if self.pred_proba_ is None:
            z = self.thetas_[0] + np.dot(X_test, self.thetas_[1:])
            self.pred_proba_ = self._logistic_sigmoid(z)
        return self.pred_proba_

    def confusion_matrix(self, y_test, y_pred):
        tp, fn, tn, fp = 0, 0, 0, 0
        n_test, n_pred = len(y_test), len(y_pred)

        if n_test != n_pred:
            return f"OutputLengthError: Length of y_test, {n_test}, does not match y_pred, {n_pred}."
        else:
            for idx in range(n_test):
                if y_test[idx] == 1 and y_test[idx] == y_pred[idx]:
                    tp += 1
                elif y_test[idx] == 1 and y_test[idx] != y_pred[idx]:
                    fn += 1
                elif y_test[idx] == 0 and y_test[idx] == y_pred[idx]:
                    tn += 1
                elif y_test[idx] == 0 and y_test[idx] != y_pred[idx]:
                    fp += 1

        self.conf_mat_['TP'] = tp
        self.conf_mat_['FN'] = fn
        self.conf_mat_['TN'] = tn
        self.conf_mat_['FP'] = fp

        return {'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp}

    def model_eval(self, y_test, y_pred, y_pred_proba):
        if not self.conf_mat_:  # if dict() is empty
            cm = self.confusion_matrix(y_test, y_pred)
            tp = cm['TP']
            fn = cm['FN']
            tn = cm['TN']
            fp = cm['FP']
        else:
            tp = self.conf_mat_['TP']
            fn = self.conf_mat_['FN']
            tn = self.conf_mat_['TN']
            fp = self.conf_mat_['FP']

        accuracy = (tp + tn) / (tp + fn + tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)  # tpr and sensitivity
        specificity = tn / (tn + fp)
        fpr = fp / (tn + fp)  # 1 - specificity
        f1_score = 2 * (precision * recall) / (precision + recall)

        # calculating AUC ROC using a list of thresholds and np.trapz
        tpr_, fpr_ = [], []
        thresholds = np.arange(0.0, 1.0, 0.1)
        total_positives = sum(y_test)
        total_negatives = len(y_test) - total_positives

        for threshold in thresholds:
            false_positives, true_positives = 0, 0
            for idx, proba in enumerate(y_pred_proba):
                if proba >= threshold:
                    if y_test[idx] == 1:
                        true_positives += 1
                    else:
                        false_positives += 1
            tpr_.append(true_positives / total_positives)
            fpr_.append(false_positives / total_negatives)
        auc_roc = -1 * np.trapz(tpr_, fpr_)

        return {'accuracy': accuracy,
                'precision': precision,
                'recall (TPR/sensitivity)': recall,
                'specificity (TNR)': specificity,
                'false_positive_rate': fpr,
                'f1_score': f1_score,
                'auc_roc': auc_roc}


def standardize_features(df) -> pd.DataFrame:
    df = df.copy()
    var_types = dict(df.dtypes != 'O')
    numeric_vars = [var for var, is_numeric in var_types.items() if is_numeric == True]
    for var in numeric_vars:
        var_mean = df[var].mean()
        var_std = df[var].std()
        df[var] = round((df[var] - var_mean) / var_std, 10)
    return df


def auc_roc_curve(y_actual, y_pred_proba):
    """
    :param y_actual: numpy array
    :param y_pred_proba: numpy array
    :return: matplotlib figure

    References
    -----------
    https://mmuratarat.github.io/2019-10-01/how-to-compute-AUC-plot-ROC-by-hand
    """

    tpr_, fpr_ = [], []
    thresholds = np.arange(0.0, 1.0, 0.1)
    p = sum(y_actual)
    n = len(y_actual) - p

    for threshold in thresholds:
        fp, tp = 0, 0
        for idx, proba in enumerate(y_pred_proba):
            if proba >= threshold:
                if y_actual[idx] == 1:
                    tp += 1
                else:
                    fp += 1
        tpr_.append(tp/p)
        fpr_.append(fp/n)
    # calculate AUC ROC
    auc = -np.trapz(tpr_, fpr_)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_, tpr_, linestyle='--', marker='.', color='darkorange', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.4f' % auc)
    plt.legend(loc = "lower right")
    plt.savefig('AUC_example.png')
    plt.tight_layout()


def minimizing_cost_func_curve(costs, n_epoch_reached):
    x_axis = [i for i in range(n_epoch_reached)]
    plt.figure(figsize = (8, 6))
    plt.title('Minimizing Cross-Entropy Loss (Negative Log-Likelihood)')
    plt.xlabel('epoch iteration')
    plt.ylabel('cross - entropy loss')
    plt.plot(x_axis, costs, lw=1, linestyle='--')
    plt.tight_layout()
