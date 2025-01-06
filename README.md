# building-perceptron

## Définir les notions de Machine Learning & Deep Learning

### Machine Learning  

*Définition:*
L'Apprentissage Automatique, ou Machine Learning, est une sous discipline de l'Intelligence Artificielle qui a pour but de développer des algorithmes en mesure d'apprendre à partir de données, sans être explicitement programmés pour effectuer une tâche particulière.  
Ces algorithmes analysent des données d'entrée, identifient des motifs ou des structures puis utilisent ces connaissances afin de réaliser des prédictions ou des classifications sur de nouvelles données.  

Le Machine Learning se divise en trois principales catégories: 
- Apprentissage Supervisé: L'algorithme apprend à partir de données étiquetées.
- Apprentissage Non Supervisé: L'algorithme identifie des motifs ou des regroupements (clusters) au sein de données non étiquetées.
- Apprentissage par renforcement: L'algorithme apprend en interagissant avec un environnement et en recevant des récompenses ou des punitions en fonctions de ses actions.

Parmi les modèles de Machine Learning les plus connus on peut retrouver les Régressions Linéaires, les Arbres de Décision, les Fôrets Aléatoires, les SVM (Support Vector Machine) ou encore des algorithmes de clustering comme le K-Means par exemple.  

### Deep Learning  

*Définition:*  
L'Apprentissage Profond, ou Deep Learning, est une sous catégorie du Machine Learning qui utilise des réseaux de neurones artificiels composés de plusieurs couches. Ces réseaux de neurones imitent, d'une certaine manière, le fonctionnement des neurones biologiques et permettent de modéliser des relations complèxes dans les données.  

Contrairement aux algorithmes de Machine Learning classiques, le Deep Learning est particulièrement adapté aux grandes quantités de données et à des tâches complexes comme la reconnaissance d’images, le traitement du langage naturel, ou encore la traduction automatique. Il excelle dans l'extraction automatique des caractéristiques pertinentes (features) des données, réduisant ainsi la nécessité d’une étape de prétraitement ou d’ingénierie des caractéristiques.  

Les architectures des réseaux de neurones profondes comprennent:  
- Réseaux de Neurones Convolutif (CNN): utilisés dans le traitement d'image
- Réseaux de Neurones Récurrents (RNN): adaptés aux données séquentielles comme les séries temporelles ou les textes
- Transformers: architecture clé pour les modèles modernes de traitement du langage naturel. 


### Comparaison entre Machine Learning & Deep Learning  

| **Critères**                     | **Machine Learning**                                                                                          | **Deep Learning**                                                                                           |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Définition**                   | Utilisation d'algorithmes pour apprendre à partir de données et faire des prédictions ou prendre des décisions. | Branche du Machine Learning basée sur des réseaux de neurones artificiels profonds pour traiter des données complexes. |
| **Données nécessaires**          | Fonctionne avec des ensembles de données relativement petits (étiquetés ou non).                              | Nécessite de grands ensembles de données pour obtenir de bons résultats.                                   |
| **Ingénierie des caractéristiques** | Forte dépendance à l'ingénierie des caractéristiques, effectuée manuellement par des experts.                | Apprend automatiquement les caractéristiques pertinentes à partir des données brutes.                      |
| **Complexité des modèles**       | Modèles plus simples comme les régressions, les arbres de décision ou les SVM.                                | Modèles complexes avec plusieurs couches (réseaux convolutifs, récurrents, etc.).                          |
| **Temps d'entraînement**         | Plus rapide à entraîner grâce à des modèles légers.                                                           | Plus long à entraîner en raison de la profondeur des réseaux et de la puissance de calcul requise.         |
| **Puissance de calcul**          | Peut fonctionner sur des machines avec des ressources limitées.                                               | Nécessite des GPU ou des TPU pour traiter de grandes quantités de données et entraîner des modèles.         |
| **Interprétabilité**             | Modèles généralement plus interprétables et faciles à expliquer.                                              | Modèles considérés comme des "boîtes noires", difficilement interprétables.                                |
| **Exemples d'applications**      | Prévisions financières, détection de fraude, systèmes de recommandation simples.                              | Voitures autonomes, reconnaissance vocale, vision par ordinateur, traitement du langage naturel.           |


### Quand utiliser le Machine Learning ?

Le Machine Learning est préférable lorsque :

- Le volume de données est limité. Les algorithmes traditionnels peuvent donner de bons résultats avec moins de données.  
- Les ressources matérielles sont restreintes. Le Machine Learning ne nécessite pas de GPU ou d'infrastructures coûteuses. 
- Les données sont simples et structurées (comme les données tabulaires).  
- Une interprétabilité est nécessaire. Si l’explicabilité est essentielle, comme dans le domaine médical ou financier, les modèles comme les arbres de décision sont adaptés.  

Exemples d’applications :  

- Prévision des ventes d’un produit à partir de données historiques.  
- Détection de fraudes dans des transactions bancaires.  
- Classification de courriers électroniques (spam ou non).  
- Segmentation de clients pour des campagnes marketing.  

### Quand utiliser le Machine Learning ?  

Le Deep Learning est préférable lorsque :

- Le volume de données est très important. Les réseaux de neurones profonds s'améliorent avec plus de données.  
- Les données sont complexes ou non structurées (images, vidéos, audio, texte).  
- Les performances sont prioritaires sur l’interprétabilité. Dans certains cas, obtenir les meilleurs résultats est plus important que de comprendre le fonctionnement exact du modèle.  
- Des ressources matérielles importantes sont disponibles, comme des GPU ou des TPU.  

Exemples d’applications :  

- Vision par ordinateur : Reconnaissance faciale, détection d’objets dans des images.  
- Traitement du langage naturel : Chatbots, traduction automatique, analyse de sentiment.  
- Audio et parole : Reconnaissance vocale, synthèse vocale.  
- Jeux et simulations : Entraînement de modèles pour des jeux complexes comme Go ou des simulations en robotique.  

## Applications du Deep Learning  

Le Deep Learning a révolutionné de nombreux domaines grâce à sa capacité à analyser et à modéliser des données complexes. Voici trois applications majeures qui illustrent ses potentialités.  

### 1) Reconnaissance d’images et vision par ordinateur  

*Description*  

Le Deep Learning a transformé le domaine de la vision par ordinateur en permettant aux machines de "voir" et d’interpréter des images ou des vidéos. Les réseaux de neurones convolutifs (CNN) sont particulièrement adaptés à ce type de tâche, car ils détectent automatiquement les motifs pertinents dans les images, comme les contours, les textures, ou les formes.  

*Exemples d'applications*  

- Reconnaissance faciale : Utilisée dans des systèmes de sécurité pour déverrouiller des appareils (comme Face ID d’Apple) ou pour identifier des individus dans des lieux publics.  
- Diagnostic médical assisté : Détection de maladies à partir de radiographies, d’IRM ou de scanners. Par exemple, les réseaux de neurones sont utilisés pour identifier des tumeurs dans des mammographies ou pour détecter la rétinopathie diabétique.  
- Détection d’objets : Utilisée dans les voitures autonomes pour reconnaître des piétons, des véhicules ou des panneaux de signalisation.  

*Exemple concret:*  
DeepMind’s AlphaFold : Un système basé sur le Deep Learning capable de prédire la structure 3D des protéines à partir de leur séquence d’acides aminés. Cette avancée a des implications majeures pour la recherche biomédicale et le développement de nouveaux traitements.  

*Impacts*  

- Amélioration des diagnostics médicaux, souvent plus précis que les experts humains.  
- Sécurité renforcée grâce à des systèmes de reconnaissance biométrique.  
- Développement de technologies révolutionnaires comme les véhicules autonomes.  

### 2) Traitement du langage naturel (NLP)  

*Description*  
Le traitement du langage naturel vise à permettre aux machines de comprendre, générer et interagir avec le langage humain. Les architectures modernes, comme les Transformers (utilisées par des modèles comme GPT d’OpenAI ou BERT de Google), ont permis des avancées spectaculaires dans ce domaine.  

*Exemples d'applications*  
- Chatbots intelligents : Des assistants virtuels comme ChatGPT ou Google Assistant, capables de répondre à des questions complexes, d’automatiser le support client, ou même d’interagir en plusieurs langues.
- Traduction automatique : Services comme Google Traduction qui génèrent des traductions précises en tenant compte du contexte.
- Résumé automatique de documents : Utilisé pour extraire les points clés de rapports ou d’articles de manière automatique.  

*Exemple concret:*  
OpenAI’s GPT-4 : Un modèle de traitement du langage capable d’écrire des textes complexes, de traduire des langues, de coder, ou de fournir des explications détaillées dans divers domaines. Il est utilisé dans des applications éducatives, de recherche et d’assistance.  

*Impacts*  
- Simplification de la communication entre individus parlant différentes langues.  
- Accélération des processus de support client grâce à l’automatisation.  
- Aide à la prise de décision dans des domaines comme les affaires, l’éducation, et la recherche.  

### 3) IA generative  

*Description*  
Le Deep Learning a permis la naissance d’un nouveau domaine appelé intelligence artificielle générative, où les modèles créent du contenu original, comme des images, des vidéos, ou de la musique. Les architectures les plus utilisées incluent les réseaux antagonistes génératifs (GANs) et les Transformers adaptés à la génération d’images.  

*Exemples d'applications*  
- Création artistique assistée : Des outils comme DALL·E 2 d’OpenAI permettent de générer des illustrations ou des œuvres d’art à partir de descriptions textuelles.  
- Reconstruction d’images ou restauration : Utilisée pour restaurer des photos anciennes ou endommagées.  
- Création de contenu vidéo : Génération de visages humains artificiels pour des publicités, des jeux vidéo, ou des films (DeepFake).  

*Exemple concret:*  
AI Experiments – Google’s DeepDream : Un projet qui utilise des réseaux de neurones convolutifs pour générer des images surréalistes en amplifiant des motifs présents dans une image existante. Ce type de projet explore la créativité artificielle et les perceptions visuelles.  

*Impacts*  
- Transformation du secteur de la création artistique et de l’industrie du divertissement.  
- Applications pratiques dans la restauration du patrimoine visuel.  
- Défis éthiques, comme l’utilisation de DeepFakes pour manipuler des vidéos ou des images.  

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
