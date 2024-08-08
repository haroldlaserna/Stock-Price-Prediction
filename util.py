import warnings
warnings.filterwarnings('ignore')

import os
import json
from utils import plot_data, plot_all_predictions

def clear_terminal():
    """
    Limpia la pantalla del terminal.

    Esta función utiliza el método `os.system` para ejecutar el comando 'clear',
    que se usa para limpiar el contenido de la pantalla del terminal en sistemas
    operativos tipo Unix. Si estás ejecutando este script en Windows, es posible
    que necesites reemplazar 'clear' por 'cls' para que la función funcione
    correctamente.

    Ejemplo:
        >>> clear_terminal()
    """
    os.system('clear')

def load_config(filename):
    """
    Carga la configuración desde un archivo JSON.

    Esta función lee un archivo JSON especificado por `filename` y analiza su
    contenido en un diccionario de Python. El archivo JSON debe contener los
    datos de configuración necesarios para la aplicación.

    Args:
        filename (str): La ruta al archivo de configuración JSON.

    Returns:
        dict: Un diccionario que contiene los datos de configuración cargados
              desde el archivo JSON.

    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
        json.JSONDecodeError: Si el archivo no es un JSON válido.

    Ejemplo:
        >>> config = load_config('config.json')
        >>> print(config)
        {'symbol': 'AAPL', 'start_date': '2020-01-01', 'end_date': '2020-12-31'}
    """
    with open(filename, 'r') as f:
        return json.load(f)