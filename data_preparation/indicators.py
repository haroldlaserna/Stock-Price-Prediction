import pandas as pd
import numpy as np
from tensorflow.keras.metrics import MeanSquaredError
def calculate_moving_averages(data, windows):
    """
    Calcula medias móviles para diferentes ventanas.

    :param data: DataFrame con los datos históricos de la acción.
    :param windows: Lista de ventanas para las medias móviles.
    :return: DataFrame con las medias móviles añadidas.
    """
    for window in windows:
        column_name = f'MA{window}'
        data[column_name] = data['Close'].rolling(window=window).mean()
    return data
    
def evaluate_model(model, X, y):
    """
    Evalúa el modelo LSTM.

    :param model: Modelo LSTM.
    :param X: Datos de entrada.
    :param y: Etiquetas.
    :return: Error cuadrático medio del modelo.
    """
    y_pred = model.predict(X)
    mse_metric = MeanSquaredError()
    mse_metric.update_state(y, y_pred)
    
    # Obtiene el valor del MSE
    mse = mse_metric.result().numpy()
    return mse
