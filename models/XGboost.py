import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar datos
# Asegúrate de reemplazar este DataFrame con tus datos reales.
# Ejemplo:
# df = pd.read_csv('data.csv')
# Para este ejemplo, usaremos datos ficticios.

# Suponiendo que df tiene las columnas requeridas y la columna objetivo 'next_month_close'.
data = np.random.rand(100, 15)  # Datos ficticios
target = np.random.rand(100)  # Datos ficticios para 'next_month_close'
df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Year', 'Month', 'Day', 'DayOfWeek', 
                                 'DayOfYear', 'SMA_5', 'EMA_10', 'Daily_Range', 'Daily_Change', 
                                 'Price_Volume_Ratio', 'Volatility_10D'])
df['next_month_close'] = target

# Preparar datos
X = df[['Open', 'High', 'Low', 'Close', 'Year', 'Month', 'Day', 'DayOfWeek', 
        'DayOfYear', 'SMA_5', 'EMA_10', 'Daily_Range', 'Daily_Change', 
        'Price_Volume_Ratio', 'Volatility_10D']]
y = df['next_month_close']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Opcional: Visualizar características importantes
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()

