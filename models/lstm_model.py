from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    """
    Crea un modelo LSTM.

    :param input_shape: Forma de entrada para el modelo.
    :return: Modelo LSTM.
    """
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True))
    model.add(LSTM(units=100, activation='relu', return_sequences=True))
    model.add(LSTM(units=100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
