from keras.models import Sequential
from keras.layers import Conv1D, Dense
from keras import regularizers

def create_tcn_model(input_shape, num_filters=64, kernel_size=10, dilation_rates=[1, 2, 4, 8, 16]):
    """
    Crea un modelo TCN (Temporal Convolutional Network).

    :param input_shape: Forma de entrada para el modelo (dimensiones de las secuencias temporales).
    :param num_filters: Número de filtros en las capas convolucionales.
    :param kernel_size: Tamaño del kernel para las capas convolucionales.
    :param dilation_rates: Lista de tasas de dilatación para las capas convolucionales.
    :return: Modelo TCN.
    """
    model = Sequential()
    
    # Añadir capas convolucionales con dilatación
    for dilation_rate in dilation_rates:
        model.add(Conv1D(filters=num_filters, 
                         kernel_size=kernel_size,
                         padding='causal',
                         activation='relu',
                         dilation_rate=dilation_rate,
                         input_shape=input_shape))
        #model.add(Dropout(0.2))  # Añadir dropout para prevenir el sobreajuste

    # Capa densa de salida
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model