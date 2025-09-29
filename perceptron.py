import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

class Perceptron(BaseEstimator, ClassifierMixin):
    """
    Perceptron (binaire ou multi-classe) avec options:
    - learning_rate
    - L2 (penalty='l2', alpha)
    - shuffle + random_state
    - early stopping (tol sur amélioration, patience)
    - warm_start
    - multi_class: 'ovr' (One-vs-Rest) ou 'multinomial' (vrai multi-classe)
    """

    def __init__(
        self,
        max_iter=1000,
        learning_rate=0.01,
        penalty=None,             # None | 'l2'
        alpha=0.0001,             # force L2 (si penalty='l2')
        shuffle=True,
        random_state=None,
        tol=1e-4,                 # tolérance d'amélioration (pas un nb d'erreurs)
        patience=5,               # nb d'époques sans amélioration autorisées
        warm_start=False,
        with_bias=True,
        multi_class="ovr"         # 'ovr' ou 'multinomial'
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.alpha = alpha
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.patience = patience
        self.warm_start = warm_start
        self.with_bias = with_bias
        self.multi_class = multi_class

    # ---------- Utils internes ----------
    def _init_params(self, n_classes, n_features):
        rng = check_random_state(self.random_state)

        if self.multi_class == "ovr" and n_classes > 2:
            # One-vs-Rest: un classif par classe
            self.coef_ = rng.normal(scale=0.01, size=(n_classes, n_features))
            self.intercept_ = np.zeros(n_classes) if self.with_bias else None
        else:
            # binaire ou multinomial "vrai" (poids par classe)
            n_rows = n_classes if n_classes > 2 else 1
            self.coef_ = rng.normal(scale=0.01, size=(n_rows, n_features))
            self.intercept_ = np.zeros(n_rows) if self.with_bias else None

        self.errors_history_ = []
        self.loss_history_ = []

    def _apply_penalty(self):
        if self.penalty == "l2":
            self.coef_ -= self.learning_rate * self.alpha * self.coef_

    def _margin_binary(self, X):
        # Retourne marge (w·x + b) pour binaire
        m = X @ self.coef_.ravel()
        if self.with_bias:
            m = m + self.intercept_.ravel()[0]
        return m

    def _margin_multiclass(self, X):
        # scores shape (n_samples, n_classes)
        M = X @ self.coef_.T
        if self.with_bias:
            M = M + self.intercept_
        return M

    # ---------- API sklearn ----------
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        if (not self.warm_start) or (not hasattr(self, "coef_")):
            self._init_params(n_classes, n_features)

        rng = check_random_state(self.random_state)
        best_score = -np.inf
        epochs_no_improve = 0

        # Encodage binaire interne en {-1, +1} si binaire
        if n_classes == 2 and self.multi_class != "multinomial":
            y_mapped = np.where(y == self.classes_[1], 1, -1)  # classe positive = classes_[1]
        else:
            # multi-classe : garder y indexé 0..K-1
            inv_map = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([inv_map[yy] for yy in y])

        for epoch in range(self.max_iter):
            # Shuffle
            if self.shuffle:
                idx = rng.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx] if n_classes > 2 and self.multi_class == "multinomial" else (y_mapped[idx] if n_classes == 2 and self.multi_class != "multinomial" else y_idx[idx])
            else:
                X_epoch = X
                y_epoch = y if n_classes > 2 and self.multi_class == "multinomial" else (y_mapped if n_classes == 2 and self.multi_class != "multinomial" else y_idx)

            errors = 0
            loss_epoch = 0.0

            if n_classes == 2 and self.multi_class != "multinomial":
                # ----- BINAIRE (mises à jour perceptron) -----
                for xi, yi in zip(X_epoch, y_epoch):
                    margin = xi @ self.coef_.ravel()
                    if self.with_bias:
                        margin += self.intercept_.ravel()[0]
                    pred_sign = 1 if margin >= 0 else -1
                    if pred_sign != yi:
                        # update perceptron
                        self.coef_.ravel()[:] += self.learning_rate * yi * xi
                        if self.with_bias:
                            self.intercept_.ravel()[:] += self.learning_rate * yi
                        errors += 1
                        # hinge-like loss (max(0, 1 - y*m))
                        loss_epoch += max(0.0, 1.0 - yi * margin)
                    # pénalité L2
                    self._apply_penalty()

            elif self.multi_class == "ovr":
                # ----- MULTI-CLASSE OVR -----
                # Un classif binaire par classe: y in {+1, -1} pour chaque k
                for xi, yi_idx in zip(X_epoch, y_epoch):
                    # calcul des marges par classe
                    margins = self._margin_multiclass(xi[None, :]).ravel()
                    pred = np.argmax(margins)
                    if pred != yi_idx:
                        # mise à jour: + pour vraie classe, - pour prédite
                        self.coef_[yi_idx] += self.learning_rate * xi
                        self.coef_[pred]   -= self.learning_rate * xi
                        if self.with_bias:
                            self.intercept_[yi_idx] += self.learning_rate
                            self.intercept_[pred]   -= self.learning_rate
                        errors += 1
                        # perte hinge multi-classe (Crammer-Singer approx)
                        loss_epoch += max(0.0, 1.0 + margins[pred] - margins[yi_idx])
                    self._apply_penalty()

            else:
                # ----- MULTI-CLASSE "multinomial" (perceptron vrai multi) -----
                for xi, yi_idx in zip(X_epoch, y_epoch):
                    margins = self._margin_multiclass(xi[None, :]).ravel()
                    pred = np.argmax(margins)
                    if pred != yi_idx:
                        self.coef_[yi_idx] += self.learning_rate * xi
                        self.coef_[pred]   -= self.learning_rate * xi
                        if self.with_bias:
                            self.intercept_[yi_idx] += self.learning_rate
                            self.intercept_[pred]   -= self.learning_rate
                        errors += 1
                        loss_epoch += max(0.0, 1.0 + margins[pred] - margins[yi_idx])
                    self._apply_penalty()

            # Logs
            self.errors_history_.append(errors)
            self.loss_history_.append(loss_epoch / n_samples)

            # Early stopping basé sur perf de l’époque (accuracy train)
            acc = self.score(X, y)
            if acc > best_score + self.tol:
                best_score = acc
                epochs_no_improve = 0
                # on pourrait aussi snapshot les meilleurs poids
                best_coef_ = self.coef_.copy()
                best_intercept_ = None if self.intercept_ is None else self.intercept_.copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    # restaurer meilleurs poids
                    self.coef_ = best_coef_
                    if self.intercept_ is not None:
                        self.intercept_ = best_intercept_
                    break

        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["coef_"])
        X = check_array(X)
        if self.coef_.ndim == 1 or self.coef_.shape[0] == 1:
            # binaire
            margin = self._margin_binary(X)
            return margin
        else:
            # multi-classe
            return self._margin_multiclass(X)

    def predict(self, X):
        scores = self.decision_function(X)
        if np.ndim(scores) == 1:
            # binaire
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            idx = np.argmax(scores, axis=1)
            return self.classes_[idx]

    def predict_proba(self, X):
        """
        Probas non calibrées (sigmoïde binaire / softmax multi).
        Utilisez CalibratedClassifierCV si vous avez besoin de proba calibrées.
        """
        scores = self.decision_function(X)
        if np.ndim(scores) == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        else:
            # stability
            s = scores - scores.max(axis=1, keepdims=True)
            exp = np.exp(s)
            prob = exp / exp.sum(axis=1, keepdims=True)
            return prob

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
