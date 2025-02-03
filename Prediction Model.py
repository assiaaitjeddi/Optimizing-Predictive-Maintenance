import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Masking


seed = 99
random.seed(seed)
tf.random.set_seed(seed)

def create_model(timesteps, input_dim, intermediate_dim, batch_size, latent_dim, epochs, optimizer):
    # Configuration des paramètres du réseau
    timesteps = timesteps
    input_dim = input_dim
    intermediate_dim = intermediate_dim
    batch_size = batch_size
    latent_dim = latent_dim
    epochs = epochs

    # Choix de l'optimiseur
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    else:
        print("Optimiseur non implémenté")
        exit(-1)
    
    masking_value = -99.  # Valeur utilisée pour ignorer certaines données d'entrée.

    # Classe pour effectuer l'échantillonnage (Sampling)
    # Permet de générer une représentation latente `z` en utilisant (mu, sigma).
    class Sampling(keras.layers.Layer):
        """Utilise (mu, sigma) pour échantillonner z, le vecteur encodant une trajectoire moteur."""
        def call(self, inputs):
            mu, sigma = inputs
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))  # Bruit gaussien
            return mu + tf.exp(0.5 * sigma) * epsilon

    # ----------------------- ENCODEUR -----------------------
    # Entrée de l'encodeur avec une séquence temporelle.
    inputs = Input(shape=(timesteps, input_dim,), name='encoder_input')

    # Ajout d'une couche de masquage pour ignorer certaines données.
    mask = Masking(mask_value=masking_value)(inputs)

    # Encodage avec une couche LSTM bidirectionnelle.
    h = Bidirectional(LSTM(intermediate_dim))(mask)

    # Extraction des paramètres (mu, sigma) pour l'espace latent.
    mu = Dense(latent_dim)(h)
    sigma = Dense(latent_dim)(h)

    # Échantillonnage pour générer `z`.
    z = Sampling()([mu, sigma])

    # Création du modèle encodeur.
    encoder = keras.Model(inputs, [z, mu, sigma], name='encoder')
    print(encoder.summary())

    # ----------------------- RÉGRESSSEUR --------------------
    # Entrée pour la régression à partir de l'espace latent.
    reg_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_reg')

    # Couche intermédiaire pour la régression.
    reg_intermediate = Dense(200, activation='tanh')(reg_latent_inputs)

    # Sortie de la régression (valeur scalaire).
    reg_outputs = Dense(1, name='reg_output')(reg_intermediate)

    # Création du modèle de régression.
    regressor = keras.Model(reg_latent_inputs, reg_outputs, name='regressor')
    print(regressor.summary())
    # -------------------------------------------------------

    # Décoder l'espace latent (optionnel, décommenter si nécessaire).
    '''
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    h_decoded = RepeatVector(timesteps)(latent_inputs)
    h_decoded = Bidirectional(LSTM(intermediate_dim, return_sequences=True))(h_decoded)
    outputs = LSTM(input_dim, return_sequences=True)(h_decoded)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')
    print(decoder.summary())
    '''

    # -------------------- MODÈLE PRINCIPAL --------------------
    # Classe qui combine encodeur, régressseur et éventuellement décodeur.
    class RVE(keras.Model):
        def __init__(self, encoder, regressor, decoder=None, **kwargs):
            super(RVE, self).__init__(**kwargs)
            self.encoder = encoder
            self.regressor = regressor
            self.decoder = decoder

            # Suivi des différentes pertes (KL divergence, régression, reconstruction).
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
            if self.decoder is not None:
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        @property
        def metrics(self):
            # Retourne les métriques à suivre pendant l'entraînement.
            if self.decoder is not None:
                return [
                    self.total_loss_tracker,
                    self.kl_loss_tracker,
                    self.reg_loss_tracker,
                    self.reconstruction_loss_tracker
                ]
            else:
                return [
                    self.total_loss_tracker,
                    self.kl_loss_tracker,
                    self.reg_loss_tracker,
                ]

        def train_step(self, data):
            # Déroulement d'une étape d'entraînement.
            x, target_x = data
            with tf.GradientTape() as tape:
                # Calcul de la perte KL.
                z, mu, sigma = self.encoder(x)
                kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                # Perte de régression.
                reg_prediction = self.regressor(z)
                reg_loss = tf.reduce_mean(
                    keras.losses.mse(target_x, reg_prediction)
                )

                # Perte de reconstruction (si décodeur activé).
                if self.decoder is not None:
                    reconstruction = self.decoder(z)
                    reconstruction_loss = tf.reduce_mean(
                        keras.losses.mse(x, reconstruction)
                    )
                    total_loss = kl_loss + reg_loss + reconstruction_loss
                    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                else:
                    total_loss = kl_loss + reg_loss

            # Calcul des gradients et mise à jour des poids.
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # Mise à jour des métriques.
            self.total_loss_tracker.update_state(total_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.reg_loss_tracker.update_state(reg_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "reg_loss": self.reg_loss_tracker.result(),
            }

        def test_step(self, data):
            # Étape de test (similaire à train_step sans mise à jour des poids).
            x, target_x = data
            z, mu, sigma = self.encoder(x)
            kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            reg_prediction = self.regressor(z)
            reg_loss = tf.reduce_mean(
                keras.losses.mse(target_x, reg_prediction)
            )
            if self.decoder is not None:
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    keras.losses.mse(x, reconstruction)
                )
                total_loss = kl_loss + reg_loss + reconstruction_loss
            else:
                total_loss = kl_loss + reg_loss

            return {
                "loss": total_loss,
                "kl_loss": kl_loss,
                "reg_loss": reg_loss,
            }
    # -------------------------------------------------------

    # Création du modèle final.
    rve = RVE(encoder, regressor)
    rve.compile(optimizer=optimizer)

    return rve