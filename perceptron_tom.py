import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

class Perceptron():

    def __init__(self, max_iter=1000, learning_rate=0.01, tol=0.001, regularization=None):
        # Initialise les paramètres du perceptron :
        # - max_iter : nombre maximum d'itérations pour l'entraînement.
        # - learning_rate : taux d'apprentissage pour ajuster les poids.
        # - tol : seuil de tolérance pour stopper l'entraînement si les erreurs sont inférieures.
        # - regularization : type de régularisation (actuellement supporte uniquement 'l2').
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.regularization = regularization


    
    def fit(self, X, y):
   # Entraîne le perceptron sur les données d'entrée X et les étiquettes y :
        # 1. Valide que X et y ont les bonnes dimensions et ne contiennent pas de valeurs incorrectes.
        # 2. Initialise les coefficients (poids) et le biais (intercept) à zéro.
        # 3. Effectue une mise à jour des poids pour chaque exemple d'entraînement sur un nombre maximum d'itérations.
        # 4. Applique la régularisation si spécifiée (actuellement uniquement L2).
        
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.coef_ = np.zeros(X.shape[1])

        if len(self.classes_) != 2:
            raise ValueError("y doit contenir exactement deux classes.")
        if len(X.shape) != 2 or len(y.shape) != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X doit être une matrice 2D et y un vecteur 1D avec le même nombre de lignes.")
        if np.any(y == None):
            raise ValueError("Le vecteur y ne doit pas contenir de valeurs None.")
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        print(f"Example of y: {y[:5]}")
        self.intercept_ = 0

        for i in range(self.max_iter):
            self.total_errors = 0
            errors = 0

            for xi, yi in zip(X,y ):
                # Calcule la sortie linéaire et applique une fonction d'activation simple (seuil à 0).

                linear_output = np.dot(xi, self.coef_) + self.intercept_
                y_pred = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (yi - y_pred)
                if y_pred is None:
                    raise ValueError("y_pred is None. Check your activation_func or weights initialization.")
                if y_pred is None:
                    print(f"y_pred is None for xi: {xi}")
                # Met à jour les poids et le biais en fonction de l'erreur actuelle.

                self.coef_ += update * xi
                self.intercept_ += update
                 # Applique la régularisation L2 si spécifiée.
                if self.regularization == 'l2':
                    self.coef_ -= self.learning_rate * self.tol * self.coef_
            # Arrête l'entraînement si les erreurs sont inférieures au seuil de tolérance.
            if errors <= self.tol:
                break

    def predict(self, X):
        # Prédit les étiquettes pour de nouvelles données X :
        # 1. Vérifie si le modèle est déjà entraîné.
        # 2. Calcule la sortie linéaire pour chaque exemple dans X.
        # 3. Applique une fonction d'activation pour retourner les prédictions (0 ou 1).

        check_is_fitted(self)
        X = check_array(X)
        linear_output = np.dot(X, self.coef_) + self.intercept_
        return (linear_output >= 0).astype(int)

    
    def score(self, X, y):
        # Évalue la précision du modèle sur un jeu de données donné :
        # 1. Prédictions pour les données d'entrée X.
        # 2. Compare les prédictions avec les étiquettes réelles y pour calculer la précision.

        X = check_array(X, ensure_2d=True)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def predict_proba(self, X):
        # Calcule les probabilités pour chaque exemple de X :
        # 1. Calcule la sortie linéaire à partir des poids et du biais.
        # 2. Applique la fonction sigmoïde pour transformer la sortie en probabilités (valeurs entre 0 et 1).

        linear_output = np.dot(X, self.coef_) + self.intercept_

        probabilities = 1 / (1 + np.exp(-linear_output))
        return probabilities
