# **MouseMaze**

**MouseMaze** est un projet d'apprentissage par renforcement visant à entraîner un agent (la "souris") à naviguer dans un labyrinthe. L'agent doit collecter des bonbons (candies) tout en évitant des trous (holes) pour atteindre la case d'arrivée (goal). Le projet utilise l'algorithme **DQN** (Deep Q-Network) implémenté avec **stable-baselines3**, et l'environnement est construit avec **gymnasium** et **pygame** pour le rendu.

---

## **Table des matières**

- [Description du projet](#description-du-projet)
- [Schéma de récompense](#schéma-de-récompense)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Utilisation](#utilisation)
- [Entraînement](#entraînement)
- [Test et démo](#test-et-démo)
- [Dépôt et livrables](#dépôt-et-livrables)


---

## **Description du projet**

Dans **MouseMaze**, l'agent évolue dans un environnement de type labyrinthe où la carte est générée aléatoirement à chaque réinitialisation (par exemple, après une chute dans un trou, la map change). L'objectif principal de l'agent est de :

1. **Collecter tous les bonbons** (candies) qui rapportent une récompense progressive :
   - 1er bonbon : 20 points
   - 2e bonbon : 25 points
   - 3e bonbon : 30 points, etc.
   - Bonus additionnel de +100 si tous les bonbons sont collectés.
2. **Atteindre la case d'arrivée (goal)** pour obtenir un bonus final de 500 points, à condition d'avoir collecté tous les bonbons.
   - Si le goal est atteint sans avoir collecté tous les bonbons, l'agent est pénalisé de -100 points et l'épisode se termine.
3. **Éviter les trous** (holes) qui pénalisent fortement (-100 points) et terminent immédiatement l'épisode.
4. **Minimiser les déplacements inutiles** grâce à :
   - Un coût fixe par pas de -0.005.
   - Une pénalité de -0.1 pour revisiter une case déjà visitée.

---

## **Schéma de récompense**

- **Trou (Hole - valeur 1)**
  - **Récompense :** -100  
  - **Action :** Épisode terminé

- **Bonbon (Candy - valeur 2)**
  - **Récompense progressive :**
    - 1er bonbon : 20
    - 2e bonbon : 25
    - 3e bonbon : 30, etc.
  - **Bonus additionnel :** +100 si tous les bonbons sont collectés
  - **Action :** La case devient vide après la collecte

- **Arrivée / Goal (valeur 3)**
  - **Si tous les bonbons ne sont pas collectés :**
    - **Récompense :** -100  
    - **Épisode :** Terminé
  - **Si tous les bonbons sont collectés :**
    - **Récompense :** +500  
    - **Épisode :** Terminé

- **Case vide (Frozen - valeur 0)**
  - **Récompense de base :** 0

- **Pénalités additionnelles :**
  - **Coût par pas :** -0.005
  - **Pénalité pour revisiter une case :** -0.1

---

## **Structure du projet**

La structure recommandée du dépôt est la suivante :


```bash
MouseMaze_Project/
├── README.md                # Ce fichier de documentation
├── models/                  # Dossier contenant les modèles sauvegardés
│   └── dqn_mouse_maze_random.zip  (ou dossier complet)
├── src/                     # Code source du projet
│   ├── __init__.py
│   ├── mouse_maze_env.py    # Environnement (génération de maps aléatoires)
│   ├── mouse_maze_train.py  # Script d'entraînement
│   └── mouse_maze_test.py   # Script de test et démo
└── docs/                    # (Optionnel) Documentation, captures d'écran, etc.

``` 

## **Prérequis**

- **Python 3.x**
- **gymnasium**
- **stable-baselines3**
- **NumPy**
- **Pygame**

Pour installer les dépendances, exécutez :

```bash
pip install gymnasium stable-baselines3 numpy pygame
``` 

## **Utilisation**

### **Entraînement**

Pour entraîner l'agent sur des maps aléatoires (la carte change à chaque réinitialisation), lancez :

```bash
python src/mouse_maze_train.py
``` 
- L'entraînement se fait sans affichage graphique (render_mode=None) afin d'augmenter le nombre d'itérations par seconde.
- Le modèle est entraîné (par exemple, sur 1 million de timesteps) et sauvegardé dans le dossier random_map_models


## **Test et démo**

Pour tester et visualiser la performance de l'agent, lancez :

```bash
python src/mouse_maze_test.py
``` 
- L'environnement est lancé en mode render_mode="human", ce qui affiche la map et l'agent en temps réel.
- Le script charge le modèle sauvegardé dans random_map_models et affiche diverses métriques (récompense totale, taux de succès, etc.).

## **Dépôt et livrables**

    - Dépôt sous forme de fichier zip unique contenant l'ensemble du projet.
    - La structure du projet permet d'identifier clairement les différentes parties : entraînement, test, modèles, et documentation.
    - Les modèles obtenus (dqn_mouse_maze_random.zip) sont inclus dans le dossier models.