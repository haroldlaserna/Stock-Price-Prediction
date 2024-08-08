import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol, start_date, end_date):
    """
    Obtiene los datos históricos de la acción. Descarga los datos si el archivo no existe.

    :param symbol: Símbolo de la acción (por ejemplo, 'AAPL').
    :param start_date: Fecha de inicio para los datos históricos (formato 'YYYY-MM-DD').
    :param end_date: Fecha de fin para los datos históricos (formato 'YYYY-MM-DD').
    :return: DataFrame con los datos históricos de la acción.
    """
    # Crear un nombre de archivo basado en el símbolo y fechas
    filename = f"datasets/{symbol}_{start_date}_{end_date}.csv"
    
    # Verificar si el archivo ya existe
    if not os.path.isfile(filename):
        # Si el archivo no existe, descargar los datos
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date, actions=False)
        data.index = data.index.date
        data = data.reset_index().rename(columns={"index": "date"})
        data["date"] = pd.to_datetime(data["date"])
        # Guardar los datos en un archivo CSV
        data.to_csv(filename, index=False)
        print(f"Datos descargados y guardados en {filename}")
        
    else:
        # Si el archivo ya existe, leer los datos desde el archivo CSV
        data = pd.read_csv(filename)
        data["date"] = pd.to_datetime(data["date"])
        print(f"Datos cargados desde {filename}")
    
    return data

def merge_stock_data(primary_symbol, secondary_symbols, start_date, end_date):
    """
    Une los datos históricos de varias acciones en un solo DataFrame.

    :param primary_symbol: Símbolo de la acción principal (por ejemplo, 'AAPL').
    :param secondary_symbols: Lista de símbolos secundarios.
    :param start_date: Fecha de inicio para los datos históricos (formato 'YYYY-MM-DD').
    :param end_date: Fecha de fin para los datos históricos (formato 'YYYY-MM-DD').
    :return: DataFrame con los datos combinados.
    """
    # Descargar datos del símbolo principal
    primary_data = fetch_stock_data(primary_symbol, start_date, end_date)
    
    # Descargar y unir datos secundarios
    for symbol in secondary_symbols:
        secondary_data = fetch_stock_data(symbol, start_date, end_date)
        # Renombrar columnas para cada símbolo secundario
        secondary_data = secondary_data.rename(columns={
            'Open': f'open_{symbol}',
            'High': f'high_{symbol}',
            'Low': f'low_{symbol}',
            'Close': f'close_{symbol}'
        })
        # Unir los datos secundarios al DataFrame principal
        primary_data = primary_data.merge(secondary_data[['date', f'open_{symbol}', f'high_{symbol}', f'low_{symbol}', f'close_{symbol}']],
                                          on='date', 
                                          how='left')
    
    return primary_data

def next_month(fecha, months):
    """Obtiene el siguiente mes sin alterar el día. Ajusta al último día del mes si es necesario."""
    # Obtiene el primer día del mes siguiente
    primer_dia_mes_siguiente = (fecha + pd.DateOffset(months=months)).replace(day=1)
    # Obtiene el último día del mes siguiente
    ultimo_dia_mes_siguiente = primer_dia_mes_siguiente + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    
    # Ajusta el día de la fecha original
    if fecha.day > ultimo_dia_mes_siguiente.day:
        return ultimo_dia_mes_siguiente
    else:
        return primer_dia_mes_siguiente + pd.DateOffset(days=(fecha.day - 1))
        
def add_next_month_column(data,month):
    """
    Añade una columna 'next_month' al DataFrame.

    :param data: DataFrame con una columna 'date'.
    :return: DataFrame con la columna 'next_month' añadida.
    """
    data["next_month"] = data["date"].apply(lambda fecha: next_month(fecha, months=month))
    return data

def merge_next_month_close(data):
    """
    Añade una columna 'next_month_close' al DataFrame.

    :param data: DataFrame con las columnas 'date' y 'Close'.
    :return: DataFrame con la columna 'next_month_close' añadida.
    """
    df_temp = data[['date', 'Close']].rename(columns={'date': 'next_month', 'Close': 'next_month_close'})
    data = data.merge(df_temp, on="next_month", how="left")
    return data

def fetch_data(symbol, secondary_symbols, start_date, end_date, month):
    """
    Obtiene los datos históricos de la acción y añade las columnas 'next_month' y 'next_month_close'.

    :param symbol: Símbolo de la acción (por ejemplo, 'AAPL').
    :param start_date: Fecha de inicio para los datos históricos (formato 'YYYY-MM-DD').
    :param end_date: Fecha de fin para los datos históricos (formato 'YYYY-MM-DD').
    :return: DataFrame con los datos históricos de la acción y las columnas añadidas.
    """
    data = merge_stock_data(symbol, secondary_symbols, start_date, end_date)
    data = add_next_month_column(data,month)
    data = merge_next_month_close(data)
    return data

