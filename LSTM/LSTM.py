import pandas as pd
from google.colab import drive
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/Seminario_TG/Datos históricos Futuros café C EE.UU. (BD2).csv'

df = pd.read_csv(file_path, delimiter=';',decimal=',')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
df['Último'] = pd.to_numeric(df['Último'])
df = df.sort_values(by='Fecha').reset_index(drop=True)
data = df[['Fecha', 'Último']].copy()

data.set_index('Fecha', inplace=True)
dataset = data['Último'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60 

X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


model = Sequential()
model.add(LSTM(
    units=50, 
    return_sequences=True, 
    input_shape=(look_back, 1)
))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=25, batch_size=32, verbose=0)
print("Modelo LSTM entrenado con éxito.")

predicted_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(predicted_scaled)
y_true_unscaled = scaler.inverse_transform(Y_test.reshape(-1, 1))
y_train_unscaled = scaler.inverse_transform(Y_train.reshape(-1, 1))
y_train_original = y_train_unscaled.flatten()

y_true = y_true_unscaled.flatten()
y_pred = y_pred.flatten()

# --- Funciones de Métricas ---
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_rnmse_rrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_true = np.mean(y_true)
    return (rmse / mean_true) * 100

def calculate_mase(y_train, y_true, y_pred):
    mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if mae_naive == 0: return np.inf
    mae_test = mean_absolute_error(y_true, y_pred)
    return mae_test / mae_naive

def calculate_theil_u(y_true, y_pred):
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean((y_true - np.roll(y_true, 1)) ** 2))
    if denom == 0: return np.inf
    return num / denom

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = calculate_mape(y_true, y_pred)
rnmse_rrmse = calculate_rnmse_rrmse(y_true, y_pred)
mase = calculate_mase(y_train_original, y_true, y_pred)
theil_u = calculate_theil_u(y_true, y_pred)


print("\n--- Indicadores de Evaluación LSTM ---")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Error Porcentual Absoluto Medio (MAPE): {mape:.2f}%")
print(f"Raíz del Error Cuadrático Medio Normalizado (RNMSE/RRMSE): {rnmse_rrmse:.2f}%")
print(f"Error Absoluto Escalonado Medio (MASE): {mase:.4f}")
print(f"Coeficiente Theil-U (U de Theil): {theil_u:.4f}")

fechas_test = data.index[train_size + look_back:]

# Crear el DataFrame para la comparación
df_comparacion = pd.DataFrame({
    'Fecha': fechas_test,
    'Real': y_true,
    'Pronosticado': y_pred
})

df_comparacion['Error_Absoluto'] = np.abs(df_comparacion['Real'] - df_comparacion['Pronosticado'])
df_comparacion['Error_Porcentual'] = (df_comparacion['Error_Absoluto'] / df_comparacion['Real']) * 100


print("\n--- Tabla Comparativa (Real vs. Pronosticado, primeras 10 filas) ---")
print(df_comparacion.head(10).round(4))
