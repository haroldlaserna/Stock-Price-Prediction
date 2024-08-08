import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

def build_generator(latent_dim, sequence_length, num_features):
    """
    Crea el modelo generador para el GAN.
    
    :param latent_dim: Dimensión del espacio latente.
    :param sequence_length: Longitud de la secuencia temporal.
    :param num_features: Número de características en la secuencia.
    :return: Modelo generador.
    """
    model = Sequential()
    model.add(Dense(128 * sequence_length * num_features, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((sequence_length, num_features)))
    model.add(Dense(num_features, activation='tanh'))
    model.add(Reshape((sequence_length, num_features)))
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def build_discriminator(sequence_length, num_features):
    """
    Crea el modelo discriminador para el GAN.
    
    :param sequence_length: Longitud de la secuencia temporal.
    :param num_features: Número de características en la secuencia.
    :return: Modelo discriminador.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(sequence_length, num_features)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def build_gan(generator, discriminator):
    """
    Crea el modelo GAN combinando el generador y el discriminador.
    
    :param generator: Modelo generador.
    :param discriminator: Modelo discriminador.
    :return: Modelo GAN.
    """
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim, X_train):
    """
    Entrena el GAN.
    
    :param generator: Modelo generador.
    :param discriminator: Modelo discriminador.
    :param gan: Modelo GAN.
    :param epochs: Número de épocas de entrenamiento.
    :param batch_size: Tamaño del batch.
    :param latent_dim: Dimensión del espacio latente.
    :param X_train: Datos reales de entrenamiento.
    """
    for epoch in range(epochs):
        # Entrenar al discriminador
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_seqs = X_train[idx]
        fake_seqs = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_seqs, labels_real)
        d_loss_fake = discriminator.train_on_batch(fake_seqs, labels_fake)
        
        # Entrenar al generador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)
        
        # Imprimir el progreso
        print(f"{epoch}/{epochs} [D loss: {d_loss_real[0] + d_loss_fake[0]} | D accuracy: {0.5 * (d_loss_real[1] + d_loss_fake[1])}] [G loss: {g_loss}]")

# Parámetros
latent_dim = 100
sequence_length = 30  # Longitud de la secuencia temporal
num_features = 15  # Número de características en los datos
epochs = 10000
batch_size = 64

# Construir modelos
generator = build_generator(latent_dim, sequence_length, num_features)
discriminator = build_discriminator(sequence_length, num_features)
gan = build_gan(generator, discriminator)

# X_train debe tener la forma (n_samples, sequence_length, num_features)
X_train = np.random.normal(0, 1, (1000, sequence_length, num_features))  # Sustituir con tus datos reales

# Entrenar GAN
train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim, X_train)

