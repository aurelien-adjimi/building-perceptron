# building-perceptron

## Répondre aux questions suivantes:

1. Qu’est ce qu’un Perceptron ? Quel est le lien entre un neurone biologique et un perceptron ?   

Un perceptron est un modèle de neurone artificiel, basé sur le neurone biologique, qui peut être utilisé pour effectuer des tâches de classification binaire. 

Le neurone biologique reçoit des signaux électriques de plusieurs autres neurones à travers les dendrites. Ces signaux sont pondérés par des synapses et sommés dans le corps cellulaire du neurone. Si la somme dépasse un certain seuil, le neurone envoie un signal électrique le long de l'axone.

Un perceptron prend plusieurs entrées, applique des poids aux entrées, les somme et passe le résultat à travers une fonction d'activation pour produire une sortie. La sortie est 0 ou 1, qui est ensuite utilisée pour prédire la classe de l'entrée. Le perceptron est utilisé dans les réseaux de neurones pour effectuer des tâches de classification binaire.


2. Quelle est la fonction mathématique du Perceptron et son usage ? Définissez les termes de l’équation.   

La fonction mathématique du perceptron est définie comme suit :   

```python
y = f(w1*x1 + w2*x2 + ... + wn*xn)
```

Où :
- `y` est la sortie du perceptron,
- `f` est la fonction d'activation,
- `w1, w2, ..., wn` sont les poids associés aux entrées `x1, x2, ..., xn`.

La fonction d'activation est généralement une fonction seuil qui produit une sortie binaire (0 ou 1) en fonction de la somme pondérée des entrées. L'usage du perceptron est de prédire la classe de l'entrée en fonction des poids associés aux entrées et de la fonction d'activation.

3. Donnez une ou plusieurs règles d’apprentissage du Perceptron.   

Le perceptron est un modèle d'apprentissage supervisé basé sur un algorithme d'apprentissage simple.
Voici quelques règles fondamentales d'apprentissage du perceptron :   

i. Mise à jour des poids: Si le Perceptron fait une erreur dans sa prédiction pour un exemple donnée, ses poids w sont mis à jour selon la règle suivante:
```python
w(t+1) = w(t) + η(y - y_pred)x
```
Où:
- `w(t)` est le vecteur de poids à l'itération t,
- `η` est le taux d'apprentissage (learning rate),
- `y` est la vraie valeur de la classe (valeur cible),
- `y_pred` est la valeur prédite par le Perceptron,
- `x` est le vecteur d'entrée.

ii. Condition d'arrêt: L'algorithme d'apprentissage du Perceptron s'arrête lorsque tous les exemples d'entraînement sont classés correctement ou lorsque le nombre d'itérations atteint un certain seuil.

iii. Seuil de décision: La sortie du Perceptron dépend de la fonction d'activation. Typiquement :
```python
y_pred = 1 si w*x >= 0
y_pred = 0 sinon
```

4. Le perceptron utilise généralement une fonction d’activation, laquelle ?   

Le perceptron utilise généralement une fonction d'activation seuil, également appelée fonction d'activation de Heaviside. La fonction d'activation seuil produit une sortie binaire (0 ou 1) en fonction de la somme pondérée des entrées. La fonction d'activation seuil est définie comme suit :
```python
f(x) = 0 si x < 0
f(x) = 1 si x >= 0
```
La fonction d'activation seuil est utilisée pour introduire une non-linéarité dans le modèle de perceptron et pour produire une sortie binaire qui peut être utilisée pour prédire la classe de l'entrée.

5. Quel est le processus d'entraînement du Perceptron ?   

Le processus d'entraînement se déroule comme suit :   

i. Initialisation des poids :

Les poids w et le biais 𝑏 sont initialisés (souvent à zéro ou à de petites valeurs aléatoires).   

ii. Présentation des données :

Chaque exemple de l'ensemble d'entraînement est présenté au Perceptron, un par un.   

iii. Calcul de la sortie :

La sortie y^ est calculée en appliquant la fonction d'activation à 𝑤 ⋅ 𝑥 + 𝑏 .   

iv. Mise à jour des poids :

Si 𝑦^ ≠ 𝑦 (lorsqu’il y a une erreur), les poids et le biais sont mis à jour selon la règle mentionnée ci-dessus.   

v. Répétition :

L'ensemble des données est parcouru plusieurs fois (époques) jusqu'à ce que :
Les erreurs soient minimisées ou,
Le nombre maximal d'itérations soit atteint.   

vi. Convergence :

Si les données sont linéairement séparables, le Perceptron converge et trouve une solution. Si elles ne le sont pas, l'algorithme peut ne jamais s'arrêter (dans ce cas, des variations comme le Perceptron à marges ou l’utilisation du SVM sont utilisées).



6. Quelles sont les limites du Perceptron ?   

Les limites du perceptron sont les suivantes :
- Le perceptron ne peut pas apprendre des fonctions non linéaires, car il utilise une fonction d'activation linéaire.
- Le perceptron est limité à la classification binaire, car il produit une sortie binaire (0 ou 1).
- Le perceptron peut être sensible aux valeurs aberrantes dans les données d'entraînement, ce qui peut affecter sa capacité à généraliser.
- Le perceptron peut nécessiter un grand nombre d'itérations pour converger vers une solution optimale, en particulier pour des problèmes complexes.
- Le perceptron peut être sensible au choix des hyperparamètres, tels que le taux d'apprentissage et le nombre d'itérations.

7. Vous développez votre propre Perceptron à l’aide de Python en
programmation orientée objet. Vous le testez sur des données factices
générées de manière aléatoire.   
