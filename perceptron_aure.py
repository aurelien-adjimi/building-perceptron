import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=1000, learning_rate=0.01, tol=0.001, regularization=None):
        """
        Initialise le perceptron.
        Args:
            max_iter (int): Nombre maximal d'itérations.
            learning_rate (float): Taux d'apprentissage.
            tol (float): Tolérance pour l'arrêt précoce.
            regularization (str): 'l2' pour ajouter une régularisation L2 (optionnel).
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.regularization = regularization

    """
    Entraîne le perceptron sur un ensemble de données.
    - X : Matrice des caractéristiques (features).
    - y : Vecteur des labels (classes cibles).
    L'entraînement s'effectue en ajustant les poids en fonction des erreurs de prédiction.
    """
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0

        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                linear_output = np.dot(xi, self.coef_) + self.intercept_
                y_pred = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (yi - y_pred)

                self.coef_ += update * xi
                self.intercept_ += update
                if self.regularization == 'l2':
                    self.coef_ -= self.learning_rate * self.tol * self.coef_ 
                errors += int(y_pred != yi)

            if errors <= self.tol:
                break
        return self

    """
    Prédit la classe de nouveaux exemples en utilisant le modèle entraîné.
    - X : Matrice des caractéristiques des exemples à prédire.
    Retourne un vecteur contenant les prédictions pour chaque exemple.
    """
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        linear_output = np.dot(X, self.coef_) + self.intercept_
        return (linear_output >= 0).astype(int)

    """
    Retourne les probabilités prédites pour chaque classe en appliquant la fonction sigmoïde 
    (ou softmax pour un problème multi-classe) sur la sortie linéaire.
    Cela permet d'obtenir une estimation de la probabilité que chaque exemple appartienne à chaque classe.

    - X : Matrice des caractéristiques des exemples à prédire.
    Retourne un tableau avec les probabilités prédites pour chaque classe.
    Dans le cas d'un problème binaire, la fonction renvoie la probabilité d'appartenance à la classe positive (1) 
    et à la classe négative (0).
    """
    def predict_proba(self, X):
        """Retourne les probabilités avec softmax."""
        check_is_fitted(self)
        X = check_array(X)
        linear_output = np.dot(X, self.coef_) + self.intercept_
        probas = 1 / (1 + np.exp(-linear_output))
        return np.vstack([1 - probas, probas]).T

    """
    Évalue les performances du perceptron sur un ensemble de données.
    - X : Matrice des caractéristiques.
    - y : Vecteur des labels.
    Retourne la précision (accuracy) des prédictions.
    """
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
