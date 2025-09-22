import pandas as pd
from google.colab import drive
import csv
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Montar Google Drive
drive.mount('/content/drive')

# Ruta del archivo CSV en tu Drive
file_path = '/content/drive/MyDrive/Seminario_TG/Datos históricos Futuros café C EE.UU. (BD).csv'

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(file_path, delimiter=',', quotechar='"')

# Convertir el formato de la columna 'Fecha'
# Cambiar los puntos por barras y luego convertir a formato de fecha
df['Fecha'] = df['Fecha'].str.replace('.', '/')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
df.set_index('Fecha', inplace=True)

# Limpiar y convertir la columna 'Último' a un tipo de dato numérico
# La columna puede contener comas como separador de miles.
# También puede tener el símbolo de porcentaje si es otra columna.
# Aquí se asume que 'Último' solo tiene el valor decimal
df['Último'] = df['Último'].str.replace(',', '', regex=True)
df['Último'] = pd.to_numeric(df['Último'], errors='coerce')

# Eliminar filas con valores nulos que pudieron surgir del pre-procesamiento
df.dropna(subset=['Último'], inplace=True)

# Seleccionar la serie de tiempo y asegurarse de que esté ordenada
time_series = df['Último'].sort_index()

# ---------------------------------------------
# Paso 3: Aplicar el modelo SARIMA
# ---------------------------------------------

# Definir los parámetros del modelo SARIMA
# (p, d, q) y (P, D, Q, S)
# Estos son valores de ejemplo, se deben ajustar para tu serie de tiempo
# S = 12 se utiliza si la estacionalidad es anual (12 meses)
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12

print("Entrenando el modelo SARIMA...")
model = SARIMAX(time_series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Ajustar el modelo
results = model.fit(disp=False)

# Imprimir un resumen de los resultados del modelo
print(results.summary())

# ---------------------------------------------
# Paso 4: Visualizar resultados y predicciones
# ---------------------------------------------

# Realizar predicciones
# Obtener las predicciones del modelo para los datos originales
predictions = results.get_prediction(start=time_series.index[0], end=time_series.index[-1], dynamic=False)
predicted_mean = predictions.predicted_mean

# Graficar la serie de tiempo original y las predicciones
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Datos Observados')
plt.plot(predicted_mean, color='red', label='Predicciones del Modelo')
plt.title('Ajuste del Modelo SARIMA a los Datos de la Serie de Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Precio (Último)')
plt.legend()
plt.grid(True)
plt.show()

# Opcional: Predicción a futuro
# Si quieres predecir valores futuros, puedes usar lo siguiente
n_steps_future = 30  # Por ejemplo, predecir los próximos 30 días
pred_future = results.get_forecast(steps=n_steps_future)
pred_future_ci = pred_future.conf_int()

# Graficar la serie de tiempo con la predicción a futuro
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Datos Históricos')
plt.plot(pred_future.predicted_mean, label='Predicción Futura', color='green')
plt.fill_between(pred_future_ci.index,
                 pred_future_ci.iloc[:, 0],
                 pred_future_ci.iloc[:, 1], color='lightgreen', alpha=0.3, label='Intervalo de Confianza')
plt.title('Predicción SARIMA a Futuro')
plt.xlabel('Fecha')
plt.ylabel('Precio (Último)')
plt.legend()
plt.grid(True)
plt.show()
