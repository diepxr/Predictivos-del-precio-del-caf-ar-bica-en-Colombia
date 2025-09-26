import pandas as pd
from google.colab import drive
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/Seminario_TG/Datos históricos Futuros café C EE.UU. (BD2).csv'

df = pd.read_csv(file_path, delimiter=';',decimal=',')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
df['Último'] = pd.to_numeric(df['Último'])
df = df.sort_values(by='Fecha').reset_index(drop=True)
data = df[['Fecha', 'Último']].copy()

data['Dias'] = (data['Fecha'] - data['Fecha'].min()).dt.days
X = data['Dias'].values.reshape(-1, 1)
y = data['Último'].values

train_size = int(len(X) * 0.95)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
fechas_test = data['Fecha'].iloc[train_size:]

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

gpr = GaussianProcessRegressor(
    kernel=kernel, 
    alpha=1e-10, # Ruido asumido, se puede ajustar o estimar
    n_restarts_optimizer=10, 
    random_state=42
)

print("\n... Entrenando el modelo GPR ...")
gpr.fit(X_train, y_train)
print("Modelo GPR ajustado con éxito.")

y_pred, sigma = gpr.predict(X_test, return_std=True)

y_true = y_test.flatten()
y_pred = y_pred.flatten()

def calculate_mape(y_true, y_pred):
    """Calcula el Error Porcentual Absoluto Medio (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_rnmse_rrmse(y_true, y_pred):
    """Calcula el RNMSE (normalizado por la media de los valores reales)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_true = np.mean(y_true)
    return (rmse / mean_true) * 100

def calculate_mase(y_train, y_true, y_pred):
    """Calcula el Error Absoluto Escalonado Medio (MASE)."""
    # Usamos los datos de entrenamiento originales para la normalización
    mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if mae_naive == 0: return np.inf
        
    mae_test = mean_absolute_error(y_true, y_pred)
    return mae_test / mae_naive

def calculate_theil_u(y_true, y_pred):
    """Calcula el Coeficiente de Inexactitud (U de Theil)."""
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean((y_true - np.roll(y_true, 1)) ** 2))
    if denom == 0: return np.inf
        
    return num / denom

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = calculate_mape(y_true, y_pred)
rnmse_rrmse = calculate_rnmse_rrmse(y_true, y_pred)
mase = calculate_mase(y_train, y_true, y_pred)
theil_u = calculate_theil_u(y_true, y_pred)


print("\n--- Indicadores de Evaluación GPR ---")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Error Porcentual Absoluto Medio (MAPE): {mape:.2f}%")
print(f"Raíz del Error Cuadrático Medio Normalizado (RNMSE/RRMSE): {rnmse_rrmse:.2f}%")
print(f"Error Absoluto Escalonado Medio (MASE): {mase:.4f}")
print(f"Coeficiente Theil-U (U de Theil): {theil_u:.4f}")

df_comparacion = pd.DataFrame({
    'Fecha': fechas_test.reset_index(drop=True),
    'Real': y_true,
    'Pronosticado': y_pred,
    'Incertidumbre_Std': sigma # La desviación estándar de la predicción
})

df_comparacion['Error_Absoluto'] = np.abs(df_comparacion['Real'] - df_comparacion['Pronosticado'])
df_comparacion['Error_Porcentual'] = (df_comparacion['Error_Absoluto'] / df_comparacion['Real']) * 100


print("\n--- Tabla Comparativa GPR (Real vs. Pronosticado, primeras 10 filas) ---")
print(df_comparacion.head(10).round(4))
