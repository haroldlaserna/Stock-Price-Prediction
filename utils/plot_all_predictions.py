import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_all_predictions(data1, predictions, model_names, symbol, seq_length):
    """
    Genera y muestra un gráfico con precios de cierre reales y predicciones para múltiples modelos.

    :param data: DataFrame con los datos históricos y medias móviles.
    :param data1: DataFrame con datos para la predicción.
    :param predictions: Lista de arrays con las predicciones de cada modelo.
    :param model_names: Lista de nombres de modelos.
    :param symbol: Símbolo de la acción.
    """
    # Crear un subplot con una fila y una columna
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Precio de Cierre y Predicciones'),
        specs=[[{'type': 'xy'}]]  # Tipo de subplot
    )

    # Añadir la traza para el precio de cierre real
    fig.add_trace(go.Scatter(
        x=data1["date"], 
        y=data1['Close'], 
        mode='lines', 
        name='Precio de Cierre Real'
    ), row=1, col=1)

    # Añadir las trazas para las predicciones de cada modelo
    for prediction, model_name in zip(predictions, model_names):
        fig.add_trace(go.Scatter(
            x=data1.iloc[seq_length:]["next_month"], 
            y=prediction.reshape(1, -1)[0], 
            mode='lines', 
            name=f'Predicción {model_name}'
        ), row=1, col=1)

    # Configurar el diseño del gráfico
    fig.update_layout(
        title=f'Precio de Cierre y Predicciones de {symbol}',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True
    )

    # Configurar el diseño del gráfico con botones
    fig.update_layout(
        title=f'Precio de Cierre y Predicciones de {symbol}',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True,
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'Mostrar Todo',
                        'method': 'update',
                        'args': [{'visible': [True] * len(predictions) + [True]}]  # Mostrar todas las trazas
                    },
                    {
                        'label': 'Ocultar Todo',
                        'method': 'update',
                        'args': [{'visible': [False] * len(predictions) + [True]}]  # Ocultar todas las trazas de predicción
                    },
                    *[
                        {
                            'label': model_name,
                            'method': 'update',
                            'args': [{'visible': [False] * i + [True] + [False] * (len(predictions) - i - 1) + [True]}]  # Mostrar solo la traza seleccionada
                        } for i, model_name in enumerate(model_names)
                    ]
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.17,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }
        ]
    )


    # Mostrar el gráfico
    fig.show()