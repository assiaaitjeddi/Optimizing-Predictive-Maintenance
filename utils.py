import math
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback
import os


# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    """
    Ajouter une colonne RUL qui indique le nombre de cycles restants pour chaque unité.
    """
    # Obtenez le nombre total de cycles pour chaque unité
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    # Fusionner le max_cycle dans le DataFrame d'origine
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculer la Remaining Useful Life (RUL) pour chaque ligne
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # Supprimer la colonne max_cycle car elle n'est plus nécessaire
    result_frame = result_frame.drop("max_cycle", axis=1)
    
    return result_frame


def add_operating_condition(df):
    """
    Identifier les conditions opérationnelles des unités comme des variables catégoriques.
    """
    # Créer une copie du DataFrame pour éviter de modifier l'original
    df_op_cond = df.copy()
    
    # Appliquer un arrondi sur les paramètres de réglage et prendre leur valeur absolue
    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    
    # Convertir les réglages en chaînes de caractères et les concaténer pour former une condition opérationnelle
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                            df_op_cond['setting_2'].astype(str) + '_' + \
                            df_op_cond['setting_3'].astype(str)
    
    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names):
    """
    Standardiser les capteurs en fonction des conditions opérationnelles spécifiques.
    """
    # Initialisation du StandardScaler pour la mise à l'échelle des données
    scaler = StandardScaler()
    
    # Pour chaque condition opérationnelle unique dans le DataFrame d'entraînement
    for condition in df_train['op_cond'].unique():
        # Ajuster le scaler sur les données d'entraînement pour cette condition spécifique
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        
        # Appliquer la transformation (standardisation) aux données d'entraînement
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = \
            scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        
        # Appliquer la transformation aux données de test en utilisant le même scaler
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = \
            scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
    
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    """ 
    Appliquer un lissage exponentiel sur les données des capteurs.
    """
    df = df.copy()
    
    # Appliquer la moyenne pondérée exponentielle sur les capteurs pour chaque unité
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)
    
    # Créer un masque pour supprimer les premiers n_samples de chaque unité après lissage
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    
    # Générer le masque
    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    
    # Appliquer le masque pour exclure les premiers n_samples
    df = df[mask]
    
    return df


def gen_train_data(df, sequence_length, columns):
    """
    Générer des séquences temporelles à partir des données brutes pour l'entraînement.
    """
    # Extraire les valeurs des colonnes spécifiées
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 et +1 pour l'indexation en Python
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]

        
def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    """
    Wrapper pour générer des séquences de données pour un sous-ensemble d'unités.
    """
    # Si aucun unit_nr n'est spécifié, utiliser toutes les unités présentes dans le DataFrame
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
    
    # Générer les séquences pour chaque unité
    data_gen = (list(gen_train_data(df[df['unit_nr'] == unit_nr], sequence_length, columns))
                for unit_nr in unit_nrs)
    
    # Concaténer toutes les séquences générées en une seule matrice numpy
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    
    return data_array


def gen_labels(df, sequence_length, label):
    """
    Générer les étiquettes associées aux séquences d'entraînement.
    """
    # Extraire la colonne d'étiquette à prédire
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # Retourner les étiquettes associées à la dernière ligne de chaque séquence
    return data_matrix[sequence_length-1:num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    """
    Wrapper pour générer les étiquettes associées aux séquences d'entraînement
    pour toutes les unités spécifiées.
    """
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    # Génération des étiquettes pour chaque unité
    label_gen = [gen_labels(df[df['unit_nr']==unit_nr], sequence_length, label) 
                 for unit_nr in unit_nrs]
    
    # Concatenation des étiquettes générées
    label_array = np.concatenate(label_gen).astype(np.float32)
    
    return label_array


def gen_test_data(df, sequence_length, columns, mask_value):
    """
    Préparer des séquences pour les unités de test.
    """
    if df.shape[0] < sequence_length:
        # Si le nombre de lignes est inférieur à sequence_length, on remplit avec la valeur mask_value
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)  # padding
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # remplir avec les données disponibles
    else:
        # Sinon, on utilise directement les données disponibles
        data_matrix = df[columns].values
        
    # Extraire la dernière séquence possible
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    
    # Générer la séquence
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def get_data(dataset, sensors, sequence_length, alpha, threshold):
    # Charger, traiter et générer les ensembles de données pour l'entraînement, la validation et le test.

    # Correct path to data directory
    dir_path = r'/content'
    train_file = 'train_' + dataset + '.txt'
    test_file = 'test_' + dataset + '.txt'

    # Define column names
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    # Read data from files
    train = pd.read_csv(os.path.join(dir_path, train_file), sep=r'\s+', header=None, names=col_names)
    test = pd.read_csv(os.path.join(dir_path, test_file), sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv(os.path.join(dir_path, 'RUL_' + dataset + '.txt'), sep=r'\s+', header=None, names=['RemainingUsefulLife'])

    # Create RUL values according to the piece-wise target function
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)

    # Remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # Scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # Apply exponential smoothing
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    # Train-validation split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()): 
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)

        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

    # Create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']


# --------------------------------------- TRAINING CALLBACKS  ---------------------------------------
class save_latent_space_viz(Callback):
    def __init__(self, model, data, target):
        super().__init__()  # Initialize the base class
        self.model = model  # Store the model passed to the constructor
        self.data = data    # Store the data passed to the constructor
        self.target = target  # Store the target passed to the constructor
    
    @property
    def model(self):
        return self._model  # Use this method if you need to access the model
    
    @model.setter
    def model(self, value):
        self._model = value
        
    def on_train_begin(self, logs=None):
        self.best_val_loss = float('inf')  # Set a very high initial value
        
    def on_epoch_end(self, epoch, logs=None):
        # Get the encoder model from the layers of the main model
        encoder = self.model.layers[0]
        if logs.get('val_loss') < self.best_val_loss:
            self.best_val_loss = logs.get('val_loss')  # Update best validation loss
            # Visualize the latent space
            viz_latent_space(encoder, self.data, self.target, epoch, True, False)

def get_callbacks(model, data, target):
    model_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30),
        ModelCheckpoint(filepath='./checkpoints/checkpoint.weights.h5',  
                        monitor='val_loss', mode='min', verbose=1, 
                        save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir='./logs'),
        save_latent_space_viz(model, data, target)  # Correct instantiation here
    ]
    return model_callbacks


def viz_latent_space(encoder, data, targets=[], epoch='Final', save=False, show=True):
    z, _, _  = encoder.predict(data)
    plt.figure(figsize=(8, 10))
    if len(targets) > 0:
        plt.scatter(z[:, 0], z[:, 1], c=targets)
    else:
        plt.scatter(z[:, 0], z[:, 1])
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    
    if show:
        plt.show()
    
    if save:
        # Ensure the directory exists
        if not os.path.exists('./images'):
            os.makedirs('./images')  # Create the directory if it doesn't exist
        # Save the figure
        plt.savefig(f'./images/latent_space_epoch{epoch}.png')
    
    return z


# ----------------------------------------- FIND OPTIMAL LR  ----------------------------------------
# Classe LRFinder pour trouver le taux d'apprentissage optimal
class LRFinder:

    def __init__(self, model):
        """
        Initialisation de LRFinder.
        :param model: Le modèle Keras pour lequel trouver le taux d'apprentissage optimal.
        """
        self.model = model  # Le modèle Keras fourni
        self.losses = []    # Liste pour enregistrer les pertes
        self.lrs = []       # Liste pour enregistrer les taux d'apprentissage
        self.best_loss = 1e9  # Initialisation de la meilleure perte à une valeur très élevée

    def on_batch_end(self, batch, logs):
        """
        Callback appelé à la fin de chaque batch pour enregistrer la perte et ajuster le taux d'apprentissage.
        :param batch: Index du batch actuel.
        :param logs: Dictionnaire contenant les informations sur l'entraînement (ex: perte).
        """
        # Récupération du taux d'apprentissage actuel
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Récupération et enregistrement de la perte
        loss = logs['loss']
        self.losses.append(loss)

        # Arrêt de l'entraînement si la perte devient instable (NaN ou trop grande)
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        # Mise à jour de la meilleure perte enregistrée
        if loss < self.best_loss:
            self.best_loss = loss

        # Augmentation du taux d'apprentissage pour le batch suivant
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1, **kw_fit):
        """
        Recherche du taux d'apprentissage optimal en ajustant le taux à chaque batch.
        :param x_train: Données d'entraînement.
        :param y_train: Étiquettes d'entraînement.
        :param start_lr: Taux d'apprentissage initial.
        :param end_lr: Taux d'apprentissage final.
        :param batch_size: Taille des batches.
        :param epochs: Nombre d'époques d'entraînement.
        :param kw_fit: Arguments supplémentaires pour model.fit.
        """
        # Calcul du nombre total de batches et du multiplicateur de taux d'apprentissage
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # Sauvegarde des poids du modèle avant l'entraînement
        initial_weights = self.model.get_weights()

        # Sauvegarde et réglage du taux d'apprentissage initial
        original_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_lr)

        # Création du callback pour suivre les pertes et ajuster le taux d'apprentissage
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        # Entraînement du modèle
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], **kw_fit)

        # Restauration des poids et du taux d'apprentissage original
        self.model.set_weights(initial_weights)
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Trace la courbe de perte en fonction du taux d'apprentissage.
        :param n_skip_beginning: Nombre de points à ignorer au début.
        :param n_skip_end: Nombre de points à ignorer à la fin.
        :param x_scale: Échelle de l'axe x ('log' par défaut).
        """
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        """
        Trouve le meilleur taux d'apprentissage basé sur le changement de perte.
        :param sma: Nombre de batches pour une moyenne mobile simple.
        :param n_skip_beginning: Nombre de points à ignorer au début.
        :param n_skip_end: Nombre de points à ignorer à la fin.
        :return: Meilleur taux d'apprentissage.
        """
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]

# --------------------------------------------- RÉSULTATS  --------------------------------------------

# Fonction pour charger et retourner les parties encodeur et régresseur d'un modèle VRAE sauvegardé
def get_model(path):
    """
    Charge un modèle VRAE sauvegardé et extrait des couches spécifiques.

    Paramètres :
    - path (str) : Chemin du fichier du modèle sauvegardé.

    Retourne :
    - encoder (Model) : La partie encodeur du modèle VRAE.
    - regressor (Model) : La partie régresseur du modèle VRAE.
    """
    # Charger le modèle sauvegardé sans le recompiler
    saved_VRAE_model = load_model(path, compile=False)
    
    # Retourner l'encodeur (couche 1) et le régresseur (couche 2)
    return saved_VRAE_model.layers[1], saved_VRAE_model.layers[2]

# Fonction pour évaluer les performances du modèle à l'aide du RMSE et du score R2
def evaluate(y_true, y_hat, label='test'):
    """
    Évalue les prédictions du modèle en calculant le RMSE et le score R2.

    Paramètres :
    - y_true (array-like) : Valeurs réelles (vérités terrain).
    - y_hat (array-like) : Valeurs prédites par le modèle.
    - label (str) : Nom du jeu de données (par exemple, 'train' ou 'test').

    Affiche :
    - RMSE (Erreur Quadratique Moyenne) : Mesure la distance entre les prédictions et les valeurs réelles.
    - R2 (Score de Variance) : Mesure la proportion de variance expliquée par le modèle.
    """
    # Calcul de l'erreur quadratique moyenne
    mse = mean_squared_error(y_true, y_hat)
    # Racine carrée du MSE pour obtenir le RMSE
    rmse = np.sqrt(mse)
    # Calcul du score R2 pour mesurer la qualité des prédictions
    variance = r2_score(y_true, y_hat)
    # Affichage des résultats d'évaluation
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

# Fonction de scoring personnalisé avec pénalités exponentielles pour les écarts
def score(y_true, y_hat):
    """
    Calcule un score personnalisé pénalisant différemment les écarts positifs et négatifs.

    Paramètres :
    - y_true (array-like) : Valeurs réelles (vérités terrain).
    - y_hat (array-like) : Valeurs prédites par le modèle.

    Affiche :
    - Un score personnalisé qui reflète l'exactitude des prédictions avec des pénalités exponentielles.
    """
    res = 0  # Initialisation du score
    # Parcourir chaque paire de valeur réelle et prédite
    for true, hat in zip(y_true, y_hat):
        # Calculer la différence entre la valeur prédite et la valeur réelle
        subs = hat - true
        # Appliquer des pénalités exponentielles en fonction du signe de l'écart
        if subs < 0:  # Écart négatif
            res = res + np.exp(-subs / 10)[0] - 1
        else:  # Écart positif
            res = res + np.exp(subs / 13)[0] - 1
    # Afficher le score personnalisé
    print("score: ", res)

# Fonction pour évaluer et visualiser les résultats du modèle
def results(path, x_train, y_train, x_test, y_test):
    """
    Évalue le modèle VRAE sur les données d'entraînement et de test, et visualise l'espace latent.

    Paramètres :
    - path (str) : Chemin du modèle VRAE sauvegardé.
    - x_train (array-like) : Données d'entrée pour l'entraînement.
    - y_train (array-like) : Étiquettes (sorties) pour l'entraînement.
    - x_test (array-like) : Données d'entrée pour le test.
    - y_test (array-like) : Étiquettes (sorties) pour le test.

    Effectue :
    - Visualisation de l'espace latent.
    - Prédictions sur les données latentes.
    - Évaluation des performances (RMSE, R2).
    - Calcul du score personnalisé.
    """
    # Charger l'encodeur et le régresseur à partir du modèle sauvegardé
    encoder, regressor = get_model(path)
    
    # Visualiser l'espace latent pour les données d'entraînement et de test
    train_mu = viz_latent_space(encoder, x_train, y_train)  # Représentation latente des données d'entraînement
    test_mu = viz_latent_space(encoder, x_test, y_test)    # Représentation latente des données de test
    
    # Effectuer des prédictions à partir de la représentation latente
    y_hat_train = regressor.predict(train_mu)  # Prédictions pour les données d'entraînement
    y_hat_test = regressor.predict(test_mu)    # Prédictions pour les données de test
    
    # Évaluer les prédictions sur les ensembles d'entraînement et de test
    evaluate(y_train, y_hat_train, 'train')
    evaluate(y_test, y_hat_test, 'test')
    
    # Calculer le score personnalisé pour les données de test
    score(y_test, y_hat_test)