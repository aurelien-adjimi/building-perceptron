import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import accuracy_score

class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=1000, tol=0.001):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.coef_ = np.zeros((self.n_classes_, X.shape[1]))
        self.intercept_ = np.zeros(self.n_classes_)
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        for i in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                xi = xi.reshape(1, -1)
                scores = safe_sparse_dot(xi, self.coef_.T) + self.intercept_
                predicted = np.argmax(scores)
                if predicted != yi:
                    self.coef_[yi] += xi.flatten()
                    self.coef_[predicted] -= xi.flatten()
                    self.intercept_[yi] += 1
                    self.intercept_[predicted] -= 1
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        scores = safe_sparse_dot(X, self.coef_.T) + self.intercept_
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_

    def predict_proba(self, X):
        return self.decision_function(X)

    def predict_log_proba(self, X):
        probas = self.predict_proba(X)
        probas -= np.max(probas, axis=1)[:, np.newaxis]
        np.exp(probas, probas)
        probas /= np.sum(probas, axis=1)[:, np.newaxis]
        return np.log(probas)

    def get_params(self, deep=True):
        return {'max_iter': self.max_iter, 'tol': self.tol}

    def set_params(self, **params):
        self.max_iter = params['max_iter']
        self.tol = params['tol']
        return self