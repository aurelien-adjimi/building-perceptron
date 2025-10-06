v"""
perceptron_pedagogique.py
Classique Perceptron pédagogique pour la classification binaire.
"""

import numpy as np

class PerceptronSimple:
    def __init__(self, n_entrees):
        """
        Initialise le perceptron avec n_entrees + 1 poids (biais).
        Tous les poids sont mis à zéro.
        """
        self.poids = np.zeros(n_entrees + 1)

    def fonction_activation(self, x):
        """
        Fonction seuil : sortie 1 si x>=0, sinon 0.
        """
        return 1 if x >= 0 else 0

    def predire(self, entree):
        """
        Prédit la sortie pour une entrée donnée.
        """
        somme = self.poids[0] + np.dot(entree, self.poids[1:])
        return self.fonction_activation(somme)

    def entrainer(self, X, y, taux_apprentissage=0.1, max_epoques=100):
        """
        Entraîne le perceptron sur le jeu de données X, y.
        Met à jour les poids à chaque erreur.
        """
        for _ in range(max_epoques):
            erreurs = 0
            for xi, cible in zip(X, y):
                prediction = self.predire(xi)
                erreur = cible - prediction
                if erreur != 0:
                    self.poids[1:] += taux_apprentissage * erreur * xi
                    self.poids[0] += taux_apprentissage * erreur
                    erreurs += 1
            if erreurs == 0:
                break

    def afficher_poids(self):
        print(f"Poids du perceptron : {self.poids}")
