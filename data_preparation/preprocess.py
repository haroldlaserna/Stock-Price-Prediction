import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def add_features(df):
    """
    Añade características temporales al DataFrame.

    :param df: DataFrame con la columna 'date'.
    :return: DataFrame con las nuevas características temporales.
    """
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['DayOfWeek'] = df['date'].dt.weekday
    df['DayOfYear'] = df['date'].dt.dayofyear
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Daily_Range'] = df['High'] - df['Low']
    
    # Cálculo de medias móviles
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Cálculo del rango diario
    df['Daily_Range'] = df['High'] - df['Low']

    # Cálculo de volatilidad
    df['Volatility_10D'] = df['Close'].rolling(window=10).std()

    # Cálculo del cambio diario
    df['Daily_Change'] = df['Close'].diff()

    # Cálculo del precio-volumen
    df['Price_Volume_Ratio'] = df['Volume'] / df['Close']
    df.dropna(inplace=True)

    return df

def create_sequences(datax, datay, seq_length):
    """
    Crea secuencias de datos y etiquetas para entrenamiento.

    :param datax: Datos normalizados.
    :param datay: Etiquetas normalizadas.
    :param seq_length: Longitud de las secuencias.
    :return: Tuplas (X, y) donde X son las secuencias de datos y y son las etiquetas correspondientes.
    """
    xs = []
    ys = []
    for i in range(len(datax) - seq_length):
        x = datax[i:i+seq_length]
        xs.append(x)
        if datay is not None:
            y = datay[i+seq_length]
            ys.append(y)
            
    if datay is not None:
        return np.array(xs), np.array(ys)
    else:
        return np.array(xs)

def prepare_data(df, seq_length):
    """
    Prepara los datos para el entrenamiento del modelo LSTM.

    Esta función normaliza los datos y crea secuencias para las características y 
    las etiquetas. También devuelve los escaladores utilizados para la normalización.

    :param df: DataFrame con las columnas 'Open', 'High', 'Low', 'Close', 'next_month_close', y 'date'.
    :param seq_length: Longitud de la secuencia para crear datos de entrenamiento.
    :return: Tuplas (X, y, scaler_x, scaler_y) donde X e y son los datos de entrenamiento, 
             y scaler_x y scaler_y son los escaladores utilizados para normalizar los datos.
    """
    df = add_features(df)

    # Normalizar los datos
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    columns_to_exclude = ['next_month_close', 'date', 'next_month']
    df_scaled_x = scaler_x.fit_transform(df.drop(columns=columns_to_exclude))
    df_scaled_y = scaler_y.fit_transform(df[["next_month_close"]])
    
    X, y = create_sequences(df_scaled_x, df_scaled_y, seq_length)
    return X, y, scaler_x, scaler_y
    
    
def transform_data_x(df, scaler_x, seq_length):
    """
    Transforma los datos de entrada para el modelo LSTM.

    Esta función normaliza las características del DataFrame utilizando el escalador 
    proporcionado y crea secuencias de datos para el entrenamiento del modelo LSTM.

    :param df: DataFrame con las columnas 'Open', 'High', 'Low', 'Close', 'Year', 
               'Month', 'Day', 'DayOfWeek', 'DayOfYear'.
    :param scaler_x: Escalador utilizado para normalizar las características.
    :param seq_length: Longitud de las secuencias de datos a crear.
    :return: Array de secuencias de datos transformados y normalizados.
    """
    df = add_features(df)
    columns_to_exclude = ['date', 'next_month']
    df_scaled_x = scaler_x.transform(df.drop(columns=columns_to_exclude))    
    X = create_sequences(df_scaled_x, None, seq_length)
    return X
