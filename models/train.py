import tensorflow as tf
import numpy as np
import json

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo LSTM.

    :param model: Modelo LSTM.
    :param X_train: Datos de entrada para el entrenamiento.
    :param y_train: Etiquetas para el entrenamiento.
    :param X_test: Datos de entrada para la prueba.
    :param y_test: Etiquetas para la prueba.
    :param epochs: Número de épocas para el entrenamiento.
    :param batch_size: Tamaño del batch para el entrenamiento.
    :return: Modelo entrenado y historial de entrenamiento.
    """
    with open("config/training_config.json", 'r') as f:
        train_config = json.load(f)
    history = model.fit(X_train, y_train, epochs=train_config["epochs"], 
                        batch_size=train_config["batch_size"], 
                        validation_data=(X_test, y_test), 
                        verbose = train_config["verbose"])
    return model, history
