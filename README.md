# Optimisation de la Maintenance Prédictive

Ce projet utilise des techniques d'**apprentissage par renforcement (RL)** et d'**estimation de la durée de vie résiduelle (RUL)** pour optimiser les stratégies de maintenance prédictive.

## Objectifs

- Utiliser l'apprentissage par renforcement pour optimiser les calendriers de maintenance.  
- Prédire la fiabilité des équipements en analysant des données spatio-temporelles.  
- Réduire les temps d'arrêt et les coûts de maintenance tout en augmentant la durée de vie des systèmes.

## Structure du Projet

Le projet contient les éléments suivants :  

1. **`FD003.txt`** : Jeu de données utilisé pour l'entraînement et l'évaluation.  
2. **Classe `MaintenanceAgent.java`** : Implémente l'algorithme d'apprentissage par renforcement.  
3. **Classe `RULPredictor.py`** : Contient le modèle prédictif basé sur les réseaux de neurones.  
4. **`config.yaml`** : Fichier de configuration pour les hyperparamètres du modèle et les paramètres d'entraînement.

## Prérequis

Avant d'exécuter ce projet, assurez-vous d'avoir les éléments suivants installés :  

- **Python 3.10** avec les bibliothèques :  
  - TensorFlow / PyTorch  
  - Pandas  
  - NumPy  
  - Matplotlib  
- **Java 21** pour la simulation des agents de maintenance.

## Étapes pour exécuter le projet

### 1. Cloner le projet

Clonez ce dépôt Git sur votre machine locale :  
```bash  
git clone https://github.com/assiaaitjeddi/maintenance-predictive.git
```

### 2. Préparer le jeu de données

Assurez-vous que le fichier `FD003.txt` est placé dans le répertoire de travail :  
```plaintext
cycle,setting1,setting2,setting3,sensor1,sensor2,...,RUL
1,0.5,0.7,100,518.67,641.82,...,128
2,0.6,0.7,100,518.67,641.82,...,127
...
```

### 3. Configurer le modèle

Modifiez le fichier `config.yaml` si nécessaire pour ajuster les hyperparamètres :  
```yaml
learning_rate: 0.001
episodes: 500
batch_size: 32
...
```

### 4. Entraîner le modèle RUL

Lancez l'entraînement du prédicteur RUL en exécutant :  
```bash
python RULPredictor.py
```

### 5. Exécuter l'agent de maintenance

Compilez et exécutez l'agent avec Java :  
```bash
javac MaintenanceAgent.java
java MaintenanceAgent
```

### 6. Visualiser les résultats

Les résultats des prédictions et des stratégies optimisées seront affichés sous forme de graphiques et de logs dans la console. Vous pouvez également visualiser les performances à l'aide de Matplotlib :  
```bash
python visualize_results.py
