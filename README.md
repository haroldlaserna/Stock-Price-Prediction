# Sistema de Predicción de Precios de Acciones Basado en Modelos de Series temporales

**Nota:** Este proyecto está en construcción. Algunas funcionalidades pueden no estar completamente implementadas o probadas.

Este proyecto está diseñado para predecir el precio de cierre de acciones utilizando varios modelos de series temporales, incluyendo LSTM, GRU, y modelos bidireccionales. El proyecto incluye la carga de datos, la preparación de los mismos, el entrenamiento de modelos, y la evaluación de resultados.

## Estructura del Proyecto

El proyecto se organiza en varias carpetas y archivos:

- `config/`: Contiene archivos de configuración en formato JSON.
- `utils/`: Contiene funciones de utilidad como la visualización de datos.
- `prepare_data.py`: Funciones para la preparación y transformación de datos.
- `modelization.py`: Definiciones de los modelos a utilizar.
- `evaluate.py`: Funciones para la evaluación de los modelos y el análisis de resultados.
- `main.py`: El archivo principal que coordina el flujo del proyecto.

## Instalación

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git

2. **Instala las dependencias**: Asegúrate de tener 'pip' y crea un entorno virtual si lo deseas:
   
   ```bash
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   ```
   Luego, Instala las dependencias requeridas:
   
   ```bash
   pip install -r requirements.txt
  
   
## Archivos y Funciones

### `utils/plot_data.py`
Funciones para la visualización de datos y resultados.

- **`plot_all_predictions(data, predictions, model_names, symbol, seq_length)`**: Grafica las predicciones de diferentes modelos comparadas con los datos reales.

### `preparation/prepare_data.py`
Funciones para la preparación de datos.

- **`get_and_prepare_data(symbol, secondary_symbols, start_date, end_date, month)`**
  - **Descripción**: Obtiene datos históricos y los prepara para el modelo.
  - **Args**: Parámetros de símbolos, fechas y mes.
  - **Returns**: DataFrames con los datos preparados.

- **`prepare_model_data(data, seq_length)`**
  - **Descripción**: Prepara los datos para el modelo, dividiéndolos en conjuntos de entrenamiento y prueba.
  - **Args**: Datos y longitud de secuencia.
  - **Returns**: Datos de entrenamiento y prueba, y escaladores.

### `modelization/modelization.py`
Definiciones de modelos.

- **`create_lstm_model(input_shape)`**: Crea un modelo LSTM.
- **`create_gru_model(input_shape)`**: Crea un modelo GRU.
- **`create_bidirectional_lstm_model(input_shape)`**: Crea un modelo LSTM bidireccional.
- **`create_bidirectional_gru_model(input_shape)`**: Crea un modelo GRU bidireccional.
- **`create_cnn_lstm_model(input_shape)`**: Crea un modelo CNN-LSTM.

### `evaluate/evaluate.py`
Funciones para la evaluación de modelos.

- **`evaluate_and_plot_results(model, data1, scaler_x, scaler_y, seq_length, history)`**
  - **Descripción**: Evalúa el modelo y desescala las predicciones.
  - **Args**: Modelo, datos, escaladores, longitud de secuencia y historial.
  - **Returns**: Predicciones desescaladas y el historial del entrenamiento.

- **`plot_loss(history)`**
  - **Descripción**: Grafica la pérdida del modelo durante el entrenamiento y ajusta una regresión exponencial.
  - **Args**: Historial del entrenamiento.

### `main.py`
Archivo principal que coordina el flujo del proyecto.

- **`main()`**
  - **Descripción**: Carga la configuración, obtiene y prepara los datos, entrena los modelos y evalúa los resultados.
  - **Funcionalidad**: Carga de configuración, obtención de datos, preparación de datos, entrenamiento y evaluación de modelos, y visualización de resultados.

## Configuración

Todos los siguientes archivos de configuración se encuentran en la carpeta `config/`.

El archivo de configuración `initial_parameters.json` contiene seis variables, cada una con un propósito específico:

- **`symbol`**: La acción que se desea predecir.
- **`start_date`** y **`end_date`**: Las fechas de inicio y final para el rango de datos que se quieren utilizar, respectivamente.
- **`seq_length`**: La longitud de la ventana de datos en días que se usa como entrada para la predicción.
- **`month`**: El mes para el cual se desea realizar la predicción.
- **`secondary_symbols`**: Las acciones de otras empresas que se tomarán en cuenta para ayudar a predecir la acción especificada en **`symbol`**.

```json
{
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "seq_length": 20,
    "month": "1",
    "secondary_symbols": ["GOOGL", "MSFT"]
}
```
El archivo de configuración `models.json` contiene todos los modelos que se van a trabajar:

```json
{
    "LSTM": "create_lstm_model",
    "GRU": "create_gru_model",
    "Bidirectional LSTM": "create_bidirectional_lstm_model",
    "Bidirectional GRU": "create_bidirectional_gru_model",
    "CNN-LSTM": "create_cnn_lstm_model"
}
```


El archivo de configuración `training_config.json` contiene tres variables que configuran el proceso de entrenamiento de los modelos. Estas variables son:

- **`epochs`**: El número de épocas para el entrenamiento del modelo. En este caso, se especifica que el modelo debe entrenarse durante 10 épocas.
- **`batch_size`**: El tamaño del lote o batch utilizado durante el entrenamiento. Se ha configurado con un tamaño de lote de 500.
- **`verbose`**: El nivel de verbosidad para la salida del entrenamiento. Un valor de `1` indica que se mostrará información básica sobre el progreso del entrenamiento.
- 

```json
{
    "epochs": 10, 
    "batch_size": 500, 
    "verbose": 1
}
```

## Contribuciones
Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama para tu característica o corrección (git checkout -b feature/nueva-caracteristica).
3. Realiza tus cambios y haz commit (git commit -am 'Agrega nueva característica').
4. Empuja los cambios a tu fork (git push origin feature/nueva-caracteristica).
5. Crea un Pull Request.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.