from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Reshape

def create_cnn_lstm_model(input_shape):
    """
    Crea un modelo CNN-LSTM.

    :param input_shape: Forma de entrada para el modelo (dimensiones de las secuencias temporales).
    :return: Modelo CNN-LSTM.
    """
    model = Sequential()
    
    # Capa convolucional
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # Otra capa convolucional para extraer más características
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Aplanar la salida para la capa LSTM
    model.add(Reshape((-1, 128)))
    # Añadir la capa LSTM
    model.add(LSTM(units=100, activation='relu'))
    
    # Capa densa de salida
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model
