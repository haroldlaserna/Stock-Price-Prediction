import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Crea un codificador Transformer.
    
    :param inputs: La entrada para el codificador Transformer.
    :param head_size: Tamaño de la cabeza de atención.
    :param num_heads: Número de cabezas de atención.
    :param ff_dim: Dimensiones de la capa feed-forward.
    :param dropout: Tasa de dropout.
    :return: La salida del codificador Transformer.
    """
    # Atención multi-cabeza
    x = MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Capa feed-forward
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = Add()([x_ff, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x

def create_time_series_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0):
    """
    Crea un modelo Transformer para series temporales.

    :param input_shape: Forma de entrada para el modelo (dimensiones de las secuencias temporales).
    :param head_size: Tamaño de la cabeza de atención.
    :param num_heads: Número de cabezas de atención.
    :param ff_dim: Dimensiones de la capa feed-forward.
    :param num_transformer_blocks: Número de bloques Transformer.
    :param mlp_units: Unidades para las capas MLP después del Transformer.
    :param dropout: Tasa de dropout.
    :return: Modelo Transformer para series temporales.
    """
    inputs = Input(shape=input_shape)
    
    # Codificador Transformer
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Capa MLP
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    
    # Capa de salida
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    
    return model
