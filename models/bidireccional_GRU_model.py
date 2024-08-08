from keras.models import Sequential
from keras.layers import GRU, Dense, Bidirectional

def create_bidirectional_gru_model(input_shape):
    """
    Crea un modelo Bidirectional GRU.

    :param input_shape: Forma de entrada para el modelo.
    :return: Modelo Bidirectional GRU.
    """
    model = Sequential()
    model.add(Bidirectional(GRU(units=100, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(GRU(units=100, activation='relu', return_sequences=True)))
    model.add(Bidirectional(GRU(units=100, activation='relu')))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

