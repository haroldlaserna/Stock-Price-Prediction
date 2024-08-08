from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

def create_bidirectional_lstm_model(input_shape):
    """
    Crea un modelo Bidirectional LSTM.

    :param input_shape: Forma de entrada para el modelo.
    :return: Modelo Bidirectional LSTM.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(units=100, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(units=100, activation='relu')))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

