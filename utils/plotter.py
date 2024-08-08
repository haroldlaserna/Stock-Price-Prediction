import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_data(data1, y_prediccion, symbol, seq_length):
    """
    Genera y muestra un gráfico de precios de cierre, medias móviles y pérdidas del modelo.

    :param data: DataFrame con los datos históricos y medias móviles.
    :param symbol: Símbolo de la acción.
    :param loss: Lista de pérdidas durante el entrenamiento.
    :param val_loss: Lista de pérdidas de validación durante el entrenamiento.
    """
    # Crear un subplot con dos filas y dos columnas
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Precio de Cierre y Predicción', 'Pérdidas del Modelo', 'Métricas del Modelo'),
        row_heights=[0.7, 0.3],  # Ajusta el tamaño relativo de las filas
        vertical_spacing=0.15,  # Espacio entre filas
        specs=[[{'type': 'xy', 'colspan': 2}, None], [{'type': 'xy'}, {'type': 'table'}]]  # Tipo de cada subplot
    )

    # Añade la traza para el precio de cierre
    fig.add_trace(go.Scatter(
        x=data1["date"], 
        y=data1['Close'], 
        mode='lines', 
        name='Precio de Cierre'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data1.iloc[seq_length:]["next_month"], 
        y=y_prediccion.reshape(1,-1)[0], 
        mode='lines', 
        name='Predicciones'
    ), row=1, col=1)

    # Generar epochs_list con numpy
    epochs_list = np.arange(1, len(loss) + 1)

    # Añade la gráfica de pérdidas del modelo
    fig.add_trace(go.Scatter(
        x=epochs_list,
        y=np.log(loss),
        mode='lines',
        name='Loss'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=epochs_list,
        y=np.log(val_loss),
        mode='lines',
        name='Val Loss'
    ), row=2, col=1)

    # Añade las métricas como texto en el gráfico
    metrics_table = go.Table(
        header=dict(values=['Métrica', 'Valor'],
                    fill_color='#333333',
                    align='left'),
        cells=dict(values=[
            ['MSE Test', 'MSE All', 'R² Test', 'R² All'],
            [f'{mse_test:.4f}', f'{mse_all:.4f}', f'{r2_test:.4f}', f'{r2_all:.4f}']
        ],
        fill_color='black',
        align='left')
    )
    fig.add_trace(metrics_table, row=2, col=2)

    # Configura el diseño del gráfico
    fig.update_layout(
        title=f'Precio de Cierre y Predicción de {symbol}',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True
    )

    # Muestra el gráfico
    fig.show()
