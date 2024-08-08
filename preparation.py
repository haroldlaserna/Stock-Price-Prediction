from sklearn.model_selection import train_test_split
import pandas as pd
from data_preparation import prepare_data,fetch_data, evaluate_model
from sklearn.model_selection import train_test_split

def get_and_prepare_data(symbol, secondary_symbols, start_date, end_date, month):
    """
    Obtiene y prepara los datos históricos para el análisis.

    Esta función recupera los datos históricos de un símbolo principal y sus
    símbolos secundarios dentro de un rango de fechas especificado. Luego, elimina
    las columnas y filas con valores nulos para preparar los datos para el análisis
    o entrenamiento del modelo.

    Args:
        symbol (str): El símbolo principal del que se obtendrán los datos (por ejemplo, 'AAPL').
        secondary_symbols (list of str): Lista de símbolos secundarios cuyos datos se deben incluir.
        start_date (str): Fecha de inicio del rango de datos en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de finalización del rango de datos en formato 'YYYY-MM-DD'.
        month (str): Nombre del mes que se debe considerar para la recuperación de datos.

    Returns:
        tuple: Una tupla con dos elementos:
            - DataFrame: El DataFrame original con datos históricos, incluyendo columnas con valores nulos.
            - DataFrame: Una copia del DataFrame original sin la columna 'next_month_close' y sin filas con valores nulos.

    Notes:
        - La columna 'next_month_close' se elimina de la copia del DataFrame `data1`, ya que no se necesita para el análisis posterior.
        - Se asume que la función `fetch_data` se encarga de obtener los datos históricos y que el DataFrame resultante puede contener valores nulos.

    Example:
        >>> data, data1 = get_and_prepare_data('AAPL', ['GOOGL', 'MSFT'], '2020-01-01', '2020-12-31', 'January')
        >>> print(data.head())
        >>> print(data1.head())
    """
    # Obtener datos históricos
    data = fetch_data(symbol, secondary_symbols, start_date, end_date, month=month)
    # Eliminar columnas y filas con valores nulos
    data1 = data.drop(columns=["next_month_close"]).dropna().copy()
    data.dropna(inplace=True)
    return data, data1

def prepare_model_data(data, seq_length):
    """
    Prepara los datos para el entrenamiento y prueba del modelo.

    Esta función utiliza `prepare_data` para transformar los datos en un formato adecuado
    para el modelo y luego divide los datos en conjuntos de entrenamiento y prueba. Los
    datos se dividen en función de un tamaño de prueba especificado y se devuelven los datos
    escalados junto con los conjuntos de entrenamiento y prueba.

    Args:
        data (pd.DataFrame): El DataFrame que contiene los datos históricos para el modelo.
        seq_length (int): La longitud de la secuencia utilizada para preparar los datos
                          para el modelo, por ejemplo, el número de días anteriores para
                          predecir el valor futuro.

    Returns:
        tuple: Una tupla que contiene:
            - X_train (np.ndarray): Conjunto de características de entrenamiento.
            - X_test (np.ndarray): Conjunto de características de prueba.
            - y_train (np.ndarray): Conjunto de etiquetas de entrenamiento.
            - y_test (np.ndarray): Conjunto de etiquetas de prueba.
            - scaler_x (sklearn.preprocessing.StandardScaler): Escalador ajustado a las
              características de los datos.
            - scaler_y (sklearn.preprocessing.StandardScaler): Escalador ajustado a las
              etiquetas de los datos.

    Raises:
        ValueError: Si los datos no son válidos o no tienen el formato esperado.

    Ejemplo:
        >>> data = pd.DataFrame({'Open': [1, 2, 3], 'Close': [2, 3, 4]})
        >>> seq_length = 2
        >>> X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_model_data(data, seq_length)
        >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    """
    X, y, scaler_x, scaler_y = prepare_data(data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
    return X_train, X_test, y_train, y_test, scaler_x, scaler_y