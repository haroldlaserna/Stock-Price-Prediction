import numpy as np
from data_preparation import transform_data_x
from models import predict
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def transform_and_evaluate_predictions(model, data1, scaler_x, scaler_y, seq_length, history):
    """
    Evalúa el modelo y produce las predicciones, luego devuelve las predicciones desescaladas.

    Esta función transforma los datos de entrada utilizando el escalador `scaler_x` y realiza
    predicciones utilizando el modelo proporcionado. Luego, desescala las predicciones utilizando
    el escalador `scaler_y` y devuelve las predicciones desescaladas junto con el historial del
    entrenamiento del modelo.

    Args:
        model (keras.Model): El modelo entrenado para hacer predicciones.
        data1 (pd.DataFrame): DataFrame que contiene los datos de entrada para la evaluación del modelo.
        scaler_x (sklearn.preprocessing.StandardScaler): Escalador ajustado a las características
                                                       de los datos de entrada.
        scaler_y (sklearn.preprocessing.MinMaxScaler): Escalador ajustado a las etiquetas de los datos
                                                       para la desescalación.
        seq_length (int): La longitud de la secuencia utilizada para transformar los datos.
        history (keras.callbacks.History): El historial del entrenamiento del modelo que contiene
                                           información sobre el proceso de entrenamiento.

    Returns:
        tuple: Un tuple que contiene:
            - Y_prediccion (np.ndarray): Las predicciones desescaladas realizadas por el modelo.
            - history (keras.callbacks.History): El historial del entrenamiento del modelo, que incluye
              información sobre las métricas y pérdidas durante el entrenamiento.

    Raises:
        ValueError: Si `data1` no tiene el formato esperado o si los escaladores no están ajustados correctamente.

    Ejemplo:
        >>> model = ...  # Un modelo entrenado
        >>> data1 = pd.DataFrame({'Open': [1, 2, 3], 'Close': [2, 3, 4]})
        >>> scaler_x = StandardScaler()
        >>> scaler_y = MinMaxScaler()
        >>> seq_length = 10
        >>> history = ...  # Historial del entrenamiento
        >>> Y_prediccion, history = evaluate_and_plot_results(model, data1, scaler_x, scaler_y, seq_length, history)
        >>> print(Y_prediccion)
    """
    # Transformar datos y desescalar
    X = transform_data_x(data1, scaler_x, seq_length=seq_length)
    scaler_min = scaler_y.data_min_[0]
    scaler_max = scaler_y.data_max_[0]
    y_test_pred_2d = predict(model, X).reshape(-1, 1)
    Y_prediccion = y_test_pred_2d * (scaler_max - scaler_min) + scaler_min
    
    return Y_prediccion, history

def exp_func(x, a, b, c):
    """
    Calcula el valor de una función exponencial ajustada a los parámetros dados.

    Esta función evalúa una función exponencial de la forma:
    
        f(x) = a * exp(-b * x) + c

    donde `a`, `b`, y `c` son parámetros que determinan la forma de la función exponencial,
    y `x` es el valor o array de valores en los que se evalúa la función.

    Args:
        x (float, np.ndarray): El valor o array de valores en los que se evalúa la función exponencial.
        a (float): Parámetro de escala que multiplica la función exponencial.
        b (float): Parámetro que controla la rapidez de la decaída exponencial.
        c (float): Parámetro de desplazamiento vertical que se suma al resultado de la función exponencial.

    Returns:
        float, np.ndarray: El valor o array de valores resultantes de aplicar la función exponencial a `x`.

    Raises:
        TypeError: Si `x`, `a`, `b`, o `c` no son del tipo esperado (float o np.ndarray).

    Ejemplo:
        >>> exp_func(1.0, 2.0, 0.5, 1.0)
        1.5
        >>> exp_func(np.array([1.0, 2.0, 3.0]), 2.0, 0.5, 1.0)
        array([1.5       , 1.07808815, 0.91599406])
    """
    return a * np.exp(-b * x) + c

def plot_loss(history):
    """
    Grafica el logaritmo de pérdida del modelo durante el entrenamiento y ajusta una regresión exponencial a los datos de pérdida.

    Esta función genera una gráfica que muestra la evolución de la pérdida (`loss`) y la pérdida de validación (`val_loss`)
    a lo largo de las épocas del entrenamiento del modelo. También ajusta y muestra una curva de regresión exponencial
    sobre los valores de pérdida para visualizar la tendencia general.

    Args:
        history (keras.callbacks.History): Historial del entrenamiento del modelo que contiene los valores de pérdida
                                           y pérdida de validación por época.

    Returns:
        None: La función no devuelve ningún valor, pero muestra una gráfica de la pérdida y la regresión exponencial.

    Raises:
        ValueError: Si el historial del entrenamiento no contiene las claves esperadas 'loss' o 'val_loss'.

    Ejemplo:
        >>> history = model.fit(...)  # Ejemplo de ajuste del modelo que genera un historial
        >>> plot_loss(history)
    """
    epochs = np.arange(len(history.history['loss'])).reshape(-1, 1)
    loss_values = np.array(np.log(history.history['loss'])).reshape(-1, 1)
    val_loss_values = np.array(np.log(history.history['val_loss'])).reshape(-1, 1)
    # Ajustar el modelo
    params, _ = curve_fit(exp_func, epochs.flatten(), loss_values.flatten())

    # Predicciones
    loss_pred = exp_func(epochs, *params)

    plt.plot(loss_values, label='loss')
    plt.plot(epochs, loss_pred, label='Regresión Exponencial', linestyle='--')
    plt.plot(val_loss_values, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylim(loss_values.min()-1,loss_values.max()+1)
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()