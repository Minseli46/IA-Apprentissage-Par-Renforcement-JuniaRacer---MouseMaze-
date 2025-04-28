# JuniaRacer

**JuniaRacer** est un projet d'apprentissage par renforcement visant à entraîner un agent de course à naviguer sur un circuit simulé. L'agent apprend grâce à un algorithme de **Q-learning discret** en exploitant une Q-table et en recevant des récompenses basées sur sa performance sur le circuit. Ce projet se décline en deux modes principaux : l'entraînement et le test.

---

## Table des matières

- [Description du projet](#description-du-projet)
- [Méthodologie d'apprentissage](#méthodologie-dapprentissage)
- [Système de récompenses](#système-de-récompenses)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation et lancement](#installation-et-lancement)
  - [Entraînement](#entraînement)
  - [Test de l'agent](#test-de-lagent)
- [Personnalisation et extensions](#personnalisation-et-extensions)

---

## Description du projet

**JuniaRacer** consiste à entraîner un agent autonome à conduire une voiture dans un environnement simulé sous **Pygame**. L'agent utilise des capteurs virtuels qui mesurent les distances dans différentes directions pour éviter les collisions et rester centré sur la piste. Le driver de l'agent est défini dans le fichier `drivers/c3po.py` et la mise à jour de la Q-table est effectuée à chaque étape de l'entraînement.

L'objectif est d'obtenir un agent qui, une fois entraîné, exploite pleinement les connaissances acquises pour sélectionner les meilleures actions (via une stratégie argmax) afin de parcourir le circuit efficacement et sans collision.

---

## Méthodologie d'apprentissage

Le projet s'appuie sur la méthode du **Q-learning discret** :

- **Q-table :**  
  Une table de dimensions adaptées au nombre d'états (définis par les distances mesurées par 5 capteurs, la vitesse et l'accélération) et au nombre d'actions possibles (freiner, accélérer, tourner à gauche ou à droite, ou ne rien faire).

- **Mise à jour de la Q-table :**  
  À chaque action, la Q-table est mise à jour selon l'équation : 

  new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)

où :
- **ALPHA** est le taux d'apprentissage,
- **GAMMA** est le facteur de discount,
- **reward** est la récompense reçue après l'action.

- **Stratégie epsilon-greedy :**  
L'agent sélectionne ses actions en combinant exploitation (choix de l'action avec la meilleure valeur) et exploration (choix aléatoire) avec une probabilité définie par **EPSILON**. En mode test, pour exploiter exclusivement les actions apprises, **EPSILON** peut être fixé à 0.

---

## Système de récompenses

Le système de récompense est conçu pour encourager l'agent à :

- **Avancer rapidement et parcourir une grande distance :**  
Une récompense proportionnelle à la vitesse (multipliée par 2) et à la distance parcourue (multipliée par 0.01).

- **Rester centré sur la piste :**  
Un bonus de centrage est calculé en fonction de la différence entre deux capteurs latéraux (d4 et d5).

- **Assurer la sécurité :**  
- Un bonus de sécurité frontale est appliqué si la distance du capteur frontal (d1) est supérieure à un seuil ; sinon, une pénalité importante (-50) est appliquée.
- Un bonus de sécurité diagonale est attribué si la distance mesurée par les capteurs diagonaux (d2 et d3) est satisfaisante, sinon une pénalité (-30) est appliquée.

- **Éviter les collisions :**  
En cas de collision, une pénalité sévère (-1000) est appliquée, et la position de l'agent est réinitialisée.

---

## Structure du projet

```bash
JuniaRacer/
├── README.md                   # Ce fichier de documentation
├── drivers/
│   └── c3po.py                 # Driver Q-learning (définit actions, mise à jour et sauvegarde de la Q-table)
├── Images/
│   └── Sprites/
│       └── white_small.png     # Sprite de la voiture
├── bg73.png                    # Image de fond représentant le circuit
├── bg43.png                    # Image utilisée pour détecter les collisions (mask)
├── JuniaRacerTrain.py          # Script d'entraînement en mode headless (sans affichage graphique)
├── JuniaRacerTest.py           # Script de test avec affichage graphique de l'agent en action
└── q_table.npy2                # Fichier de sauvegarde de la Q-table (généré après entraînement)

```

## **Prérequis**

- **Python 3.x**
- **NumPy**
- **Pygame**

Pour installer les dépendances, exécutez :

```bash
pip install numpy pygame
``` 

## Installation et lancement


 **Entraînement**

Préparation :

Assurez-vous que les fichiers bg73.png, bg43.png et Images/Sprites/white_small.png se trouvent dans les répertoires indiqués.

Lancer l'entraînement :

Le script d'entraînement (JuniaRacerTrain.py) exécute le Q-learning en mode headless pour accélérer le processus (aucun rendu graphique n'est affiché).
    
```bash
python JuniaRacerTrain.py

``` 
-   La Q-table est sauvegardée dans le fichier q_table.npy2 à la fin de l'entraînement.
-   Pendant l'entraînement, les paramètres d'exploration (EPSILON) décroissent progressivement.


**Test de l'agent**

Préparation :

Après l'entraînement, vérifiez que le fichier q_table.npy2 est présent dans le dossier racine.

Lancer le test :
Exécutez le script de test (JuniaRacerTest.py) pour observer l'agent en action avec affichage graphique.
  
```bash
python JuniaRacerTest.py

``` 
-   Le script charge la Q-table sauvegardée et fixe EPSILON = 0 et ALPHA = 0 pour désactiver l'exploration et l'apprentissage en cours de test.
-   Vous verrez la voiture se déplacer sur le circuit défini par bg73.png et utiliser bg43.png pour les détections de collision.


## Conclusion

JuniaRacer est un projet complet qui démontre l'application du Q-learning discret dans un environnement de course simulé. Que ce soit pour étudier les mécanismes de l'apprentissage par renforcement ou pour développer un agent de conduite autonome, ce projet offre une base solide pour explorer, tester et améliorer vos propres algorithmes d'apprentissage.