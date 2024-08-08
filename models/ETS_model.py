import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Cargar datos
# Asegúrate de reemplazar este DataFrame con tus datos reales.
# Ejemplo:
# df = pd.read_csv('data.csv')
# Para este ejemplo, usaremos datos ficticios.

# Suponiendo que df tiene una columna 'Close' con los precios de cierre.
data = np.random.rand(100)  # Datos ficticios de precios de cierre
df = pd.DataFrame(data, columns=['Close'])

# Preparar los datos
y = df['Close']

# Dividir en entrenamiento y prueba
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]

# Ajustar el modelo ETS
# Para suavizado exponencial simple, doble o triple
model = sm.tsa.ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Realizar predicciones
forecast = model_fit.forecast(steps=len(test))

# Evaluar el modelo
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse}")

# Visualizar resultados
plt.figure(figsize=(10, 6))
plt.plot(df.index, y, label='Datos Reales')
plt.plot(range(train_size, len(df)), forecast, color='red', label='Predicciones ETS')
plt.legend()
plt.show()
