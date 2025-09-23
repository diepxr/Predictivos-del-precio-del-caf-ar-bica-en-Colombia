import pandas as pd
from google.colab import drive
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/Seminario_TG/Datos históricos Futuros café C EE.UU. (BD2).csv'

df = pd.read_csv(file_path, delimiter=';',decimal=',')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
df['Último'] = pd.to_numeric(df['Último'])
df = df.sort_values(by='Fecha').reset_index(drop=True)
data = df[['Fecha', 'Último']].copy()

n_lags = 5
for i in range(1, n_lags + 1):
    data[f'Último_t-{i}'] = data['Último'].shift(i)
data = data.dropna()

y = data['Último']
X = data[[f'Último_t-{i}' for i in range(1, n_lags + 1)]]

split_point = len(data) - 50
X_train = X[:split_point]
y_train = y[:split_point]
X_test = X[split_point:]
y_test = y[split_point:]

print(f"\nConjunto de entrenamiento: {len(X_train)} filas")
print(f"Conjunto de prueba: {len(X_test)} filas")

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"\nError Cuadrático Medio (MSE): {mse:.4f}")
y_real = y_test
y_pred = predictions
mae = mean_absolute_error(y_real, y_pred)
print(f"\nError Absoluto Medio (MAE): {mae:.4f}")
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
print(f"Error Cuadrático Medio (RMSE): {rmse:.4f}")
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
print(f"Error Porcentual Absoluto Medio (MAPE): {mape:.4f}%")
rrmse = (rmse / np.mean(y_real)) * 100
print(f"Error Cuadrático Medio Relativo (RRMSE): {rrmse:.4f}%")
range_y = np.max(y_real) - np.min(y_real)
rnmse = (rmse / range_y) * 100
print(f"Error Cuadrático Medio Normalizado Relativo (RNMSE): {rnmse:.4f}%")
rnmse = (rmse / range_y) * 100
print(f"Error Cuadrático Medio Normalizado Relativo (RNMSE): {rnmse:.4f}%")
y_train_diff = np.abs(np.diff(y_train))
mase = mae / np.mean(y_train_diff)
print(f"Error Absoluto Escalado Medio (MASE): {mase:.4f}")
y_pred_diff = np.diff(y_pred)
y_real_diff = np.diff(y_real)
theil_u = np.sqrt(np.sum((y_pred_diff - y_real_diff)**2)) / np.sqrt(np.sum(y_real_diff**2))
print(f"Coeficiente Theil-U: {theil_u:.4f}")

print("\nValores Reales vs. Predicciones:")
results = pd.DataFrame({'Fecha': data['Fecha'][split_point:], 'Real': y_test, 'Predicción': predictions})
print(results)


#print(data.head())
