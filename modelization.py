from models import (create_lstm_model, create_gru_model, create_bidirectional_lstm_model,
                    create_bidirectional_gru_model, create_cnn_lstm_model, train_model)
import json

def create_and_train_model(create_model_func, X_train, y_train, X_test, y_test):
    """
    Crea y entrena un modelo utilizando una función de creación de modelo proporcionada.

    Esta función recibe una función para crear el modelo y los datos de entrenamiento y prueba.
    Utiliza la función proporcionada para crear el modelo y luego lo entrena con los datos de
    entrenamiento y prueba. La función devuelve el modelo entrenado y el historial del entrenamiento.

    Args:
        create_model_func (callable): Una función que crea un modelo y acepta la forma de entrada
                                       como argumento. Debe devolver una instancia de modelo.
        X_train (np.ndarray): Conjunto de características de entrenamiento.
        y_train (np.ndarray): Conjunto de etiquetas de entrenamiento.
        X_test (np.ndarray): Conjunto de características de prueba.
        y_test (np.ndarray): Conjunto de etiquetas de prueba.

    Returns:
        tuple: Un tuple que contiene:
            - model (keras.Model): El modelo entrenado.
            - history (keras.callbacks.History): El historial del entrenamiento, que contiene
              información sobre el proceso de entrenamiento, como las métricas y pérdidas durante
              el entrenamiento y la validación.

    Raises:
        TypeError: Si `create_model_func` no es una función o no acepta los argumentos esperados.
        ValueError: Si los datos de entrada no tienen la forma esperada o son incompatibles.

    Ejemplo:
        >>> def create_lstm_model(input_shape):
        >>>     model = Sequential()
        >>>     model.add(LSTM(50, input_shape=input_shape))
        >>>     model.add(Dense(1))
        >>>     model.compile(optimizer='adam', loss='mean_squared_error')
        >>>     return model
        >>>
        >>> X_train = np.random.rand(100, 10, 5)
        >>> y_train = np.random.rand(100, 1)
        >>> X_test = np.random.rand(20, 10, 5)
        >>> y_test = np.random.rand(20, 1)
        >>> model, history = create_and_train_model(create_lstm_model, X_train, y_train, X_test, y_test)
        >>> print(model.summary())
    """
    model = create_model_func((X_train.shape[1], X_train.shape[2]))
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    return model, history

def load_models_from_config(config_path):
    """
    Carga y retorna un diccionario de funciones de creación de modelos desde un archivo de configuración.

    Esta función lee un archivo de configuración en formato JSON especificado por `config_path`.
    El archivo debe contener un mapeo de nombres de modelos a nombres de funciones. Utiliza
    `globals()` para buscar y recuperar las funciones de creación de modelos correspondientes
    a los nombres especificados en el archivo JSON.

    Args:
        config_path (str): La ruta al archivo JSON que contiene la configuración de los modelos.
                           El archivo JSON debe tener la siguiente estructura:
                           {
                               "nombre_modelo1": "nombre_función1",
                               "nombre_modelo2": "nombre_función2",
                               ...
                           }
                           donde "nombre_función" es el nombre de la función en el ámbito global.

    Returns:
        dict: Un diccionario donde las claves son los nombres de los modelos y los valores son
              las funciones de creación de modelos correspondientes recuperadas mediante
              `globals()`. Si una función no se encuentra en el ámbito global, su valor en el
              diccionario será `None`.

    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
        json.JSONDecodeError: Si el archivo no es un JSON válido o tiene un formato incorrecto.

    Ejemplo:
        >>> config_path = 'models_config.json'
        >>> models = load_models_from_config(config_path)
        >>> print(models)
        {'LSTM': <function create_lstm_model at 0x...>, 'GRU': <function create_gru_model at 0x...>}
    """
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Usar globals() para obtener funciones por nombre
    models = {name: globals().get(func_name) for name, func_name in model_config.items()}
    return models