import pandas as pd
from google.colab import drive
import csv
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/Seminario_TG/Datos históricos Futuros café C EE.UU. (BD2).csv'

df = pd.read_csv(file_path, delimiter=';',decimal=',')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
df['Último'] = pd.to_numeric(df['Último'])
df = df.sort_values(by='Fecha').reset_index(drop=True)
data = df[['Fecha', 'Último']].copy()

data.set_index('Fecha', inplace=True)
y = data['Último'] 
train_size = int(len(y) * 0.98)
train, test = y[:train_size], y[train_size:]
print(f"Total de datos: {len(y)}")
print(f"Datos de entrenamiento: {len(train)}")
print(f"Datos de prueba (Test): {len(test)}")

order = (1, 1, 1)        
seasonal_order = (1, 1, 0, 12)
model = SARIMAX(
    train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
print("\n... Entrenando el modelo SARIMA ...")
results = model.fit(disp=False)
print("Modelo SARIMA ajustado con éxito.")

start = len(train)
end = len(y) - 1
predictions = results.predict(start=start, end=end, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, predictions)

print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")

y_true = test.values
y_pred = predictions.values
y_train = train.values

def calculate_mape(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = calculate_mape(y_true, y_pred)


def calculate_rnmse_rrmse(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_true = np.mean(y_true)
    return (rmse / mean_true) * 100 

rnmse_rrmse = calculate_rnmse_rrmse(y_true, y_pred)


def calculate_mase(y_train, y_true, y_pred):

    n = len(y_train)
    
    mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if mae_naive == 0:
        return np.inf  
        
    mae_test = mean_absolute_error(y_true, y_pred)
    return mae_test / mae_naive

mase = calculate_mase(y_train, y_true, y_pred)

def calculate_theil_u(y_true, y_pred):
    
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))

    denom = np.sqrt(np.mean((y_true - np.roll(y_true, 1)) ** 2))
    
    if denom == 0:
        return np.inf
        
    return num / denom

theil_u = calculate_theil_u(y_true, y_pred)


print(f"Error Porcentual Absoluto Medio (MAPE): {mape:.2f}%")
print(f"Raíz del Error Cuadrático Medio Normalizado (RNMSE/RRMSE): {rnmse_rrmse:.2f}%")
print(f"Error Absoluto Escalonado Medio (MASE): {mase:.2f}")
print(f"Coeficiente Theil-U (U de Theil): {theil_u:.2f}")

df_comparacion = pd.DataFrame({
    'Fecha': test.index,  
    'Real': test.values,  #
    'Pronosticado': predictions.values 
})
df_comparacion['Error_Absoluto'] = np.abs(df_comparacion['Real'] - df_comparacion['Pronosticado'])
df_comparacion['Error_Porcentual'] = (df_comparacion['Error_Absoluto'] / df_comparacion['Real']) * 100
print("\n--- Tabla Comparativa (Real vs. Pronosticado) ---")
print(df_comparacion.head(10).round(4))
ruta_comparacion = './comparativa_sarima_test.csv'
df_comparacion.to_csv(ruta_comparacion, index=False)
print(f"\nTabla comparativa completa guardada en: {ruta_comparacion}")
