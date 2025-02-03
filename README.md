# Optimisation de la Maintenance Prédictive

Ce projet utilise des techniques d'**apprentissage par renforcement (RL)** et d'**estimation de la durée de vie résiduelle (RUL)** pour optimiser les stratégies de maintenance prédictive.

## Objectifs

- Utiliser l'apprentissage par renforcement pour optimiser les calendriers de maintenance.  
- Prédire la fiabilité des équipements en analysant des données spatio-temporelles.  
- Réduire les temps d'arrêt et les coûts de maintenance tout en augmentant la durée de vie des systèmes.

## Structure du Projet

Le projet contient les éléments suivants :

1. **`data`** : Répertoire contenant le dataset utilisé qui est NASA C-MAPSS pour l'entraînement et l'évaluation.
2. **`EDA.ipynb`** : Notebook pour l'analyse exploratoire des données (EDA), permettant de visualiser et de mieux comprendre les caractéristiques des datasets.
3. **`Optimization Model.ipynb`** : Notebook implémentant le modèle d'optimisation basé sur l'apprentissage par renforcement.
4. **`Prediction Model.py`** : Script Python pour le modèle de prédiction de la durée de vie restante (RUL).
5. **`main.ipynb`** : Notebook principal intégrant les différentes étapes du pipeline (prétraitement, entraînement, évaluation).
6. **`utils.py`** : Script Python contenant des fonctions utilitaires utilisées dans tout le projet (prétraitement, visualisation, etc.).

## Datasets

- **NASA C-MAPSS Dataset** :
  - Données simulées "run-to-failure" pour les moteurs d'avion.
  - Conçu pour les tâches de prédiction de la RUL.

## Prérequis

Avant d'exécuter ce projet, vous devez avoir les éléments suivants installés :

- **Python 3.8+**
- **Bibliothèques Python** : NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow ou PyTorch
  

## Étapes pour exécuter le projet

### 1. Cloner le projet

Clonez ce dépôt Git sur votre machine locale :  
```bash  
git clone https://github.com/assiaaitjeddi/maintenance-predictive.git
```

### 2. Préparer le jeu de données

Assurez-vous que le dataset est placé dans le répertoire `data`.


### 3. Exécuter l'analyse exploratoire

Ouvrez et exécutez le notebook `EDA.ipynb` pour visualiser les données et comprendre leurs caractéristiques principales.

### 4. Entraîner le modèle de prédiction

Utilisez le script `Prediction Model.py` pour entraîner et évaluer le modèle RUL :
```bash
python Prediction Model.py
```

### 5. Lancer le modèle d'optimisation

Ouvrez et exécutez le notebook `Optimization Model.ipynb` pour explorer les stratégies d'optimisation basées sur le renforcement.

### 6. Exécuter le pipeline complet

Ouvrez le notebook `main.ipynb` pour exécuter toutes les étapes du projet, de la préparation des données à l'évaluation des résultats.

