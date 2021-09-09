import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_epoch=100, penalty=None, C=0.01, tolerance=1e-4, pred_threshold=0.5):
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
        self.c_m_ = dict()
        self.model_eval_metrics_ = dict()

    @staticmethod
    def _logistic_sigmoid(z):
        predict_proba = 1 / (1 + np.exp(-z))
        return predict_proba

    @classmethod
    def _cross_entropy_loss(cls, X, y, thetas):
        z = np.dot(X, thetas)
        return -1 * (np.sum(y * np.log(cls._logistic_sigmoid(z)) + (1 - y) * np.log(1 - cls._logistic_sigmoid(z))))

    @classmethod
    def _cross_entropy_loss_l1(cls, X, y, thetas, C):
        """ L1 regularization with inverse of lambda represented with C (regularization parameter).
            This approach follows sklearn's regularized cost function for logistic regression.

        References
        ----------
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        """
        z = np.dot(X, thetas)
        l1_regularization = thetas
        return C * -1 * (np.sum((y * np.log(cls._logistic_sigmoid(z))) + ((1 - y) * np.log(1 - cls._logistic_sigmoid(z)))) + l1_regularization)

    @classmethod
    def _cross_entropy_loss_l2(cls, X, y, thetas, C):
        """ L2 regularization with inverse of lambda represented with C (regularization parameter).
            This approach follows sklearn's regularized cost function for logistic regression.

        References
        ----------
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        """
        z = np.dot(X, thetas)
        l2_regularization = 0.5 * thetas.T.dot(thetas)
        return C * -1 * (np.sum((y * np.log(cls._logistic_sigmoid(z))) + ((1 - y) * np.log(1 - cls._logistic_sigmoid(z)))) + l2_regularization)

    def fit(self, X_train, y_train):
        n_rows = X_train.shape[0]
        n_features = X_train.shape[1]

        thetas = np.zeros(n_features + 1)
        tol = self.tolerance * np.ones(n_features + 1)
        X_train = np.c_[np.ones(n_rows), np.array(X_train)]

        for _ in range(self.n_epoch):
            z = np.dot(X_train, thetas)
            y_pred_proba = self._logistic_sigmoid(z)
            diffs = y_train - y_pred_proba

            if self.penalty == 'l2':
                gradient_update_terms = self.learning_rate * (self.C * np.dot(X_train, diffs) + np.sum(thetas))
            elif self.penalty == 'l1':
                gradient_update_terms = self.learning_rate * (self.C * np.dot(X_train, diffs) + np.sign(thetas))
            else:
                gradient_update_terms = self.learning_rate * np.dot(X_train, diffs)

            self.n_epoch_reached_ += 1

            if np.all(abs(gradient_update_terms) >= tol):
                # gradient descent updating step
                thetas -= gradient_update_terms
                # calculate and append cost per epoch
                if self.penalty == 'l2':
                    self.costs_.append(self._cross_entropy_loss_l2(X_train, y_train, thetas, self.C))
                elif self.penalty == 'l1':
                    self.costs_.append(self._cross_entropy_loss_l1(X_train, y_train, thetas, self.C))
                else:
                    self.costs_.append(self._cross_entropy_loss(X_train, y_train, thetas))
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
        return self

    def confusion_matrix(self, y_test, y_pred):
        TP, FN, TN, FP = 0, 0, 0, 0
        n_test, n_pred = len(y_test), len(y_pred)

        if n_test != n_pred:
            return f"OutputLengthError: Length of y_test, {n_test}, does not match y_pred, {n_pred}."
        else:
            for idx in range(n_test):
                if y_test[idx] == 1 and y_test[idx] == y_pred[idx]:
                    TP += 1
                elif y_test[idx] == 1 and y_test[idx] != y_pred[idx]:
                    FN += 1
                elif y_test[idx] == 0 and y_test[idx] == y_pred[idx]:
                    TN += 1
                elif y_test[idx] == 0 and y_test[idx] != y_pred[idx]:
                    FP += 1

        self.c_m_['TP'] = TP
        self.c_m_['FN'] = FN
        self.c_m_['TN'] = TN
        self.c_m_['FP'] = FP

        return {'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP}

    def model_eval(self, y_test, y_pred, y_pred_proba):
        if not self.c_m_:  # if dict() is empty
            cm = self.confusion_matrix(y_test, y_pred)
            tp = cm['TP']
            fn = cm['FN']
            tn = cm['TN']
            fp = cm['FP']
        else:
            tp = self.c_m_['TP']
            fn = self.c_m_['FN']
            tn = self.c_m_['TN']
            fp = self.c_m_['FP']

        accuracy = (tp + tn) / (tp + fn + tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)  # tpr and sensitivity
        specificity = tn / (tn + fp)
        fpr = fp / (fp + tn)
        f1_score = 2 * (precision * recall) /(precision + recall)

        # calculating AUC ROC using a list of thresholds and np.trapz
        tpr_, fpr_ = [], []
        thresholds = np.arange(0.0, 1.01, 0.05)
        total_positives = sum(y_test)
        total_negatives = len(y_test) - total_positives

        for threshold in thresholds:
            false_positives, true_positives = 0, 0
            for idx in range(len(y_pred_proba)):
                if y_pred_proba[idx] >= threshold:
                    if y_test[idx] == 1:
                        true_positives += 1
                    if y_test[idx] == 0:
                        false_positives += 1
            tpr_.append(true_positives/total_positives)
            fpr_.append(false_positives/total_negatives)
        auc_roc = -1 * np.trapz(tpr_, fpr_)

        return {'accuracy': accuracy,
                'precision': precision,
                'recall (TPR/sensitivity)': recall,
                'specificity (TNR)': specificity,
                'false_positive_rate': fpr,
                'f1_score': f1_score,
                'auc_roc': auc_roc}


def predicted_logistic_curve():
    pass


def minimizing_cost_curve():
    pass


def decision_boundaries_plotting():
    pass



