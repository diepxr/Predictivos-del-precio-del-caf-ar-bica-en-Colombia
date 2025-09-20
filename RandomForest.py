import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Cargar el archivo de datos
file_path = "Datos históricos Futuros café C EE.UU. (BD).csv"
df = pd.read_csv(file_path, delimiter=',')

# Mostrar las primeras filas y la información del DataFrame
print("--- Datos iniciales ---")
print(df.head())
print("\n--- Información del DataFrame ---")
print(df.info())

# 2. Preprocesamiento de los datos
# Convertir la columna de fecha a formato datetime para una mejor manipulación
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d.%m.%Y')

# Clean and convert the 'Último' column to numeric
df['Último'] = df['Último'].str.replace('"', '', regex=False).str.replace(',', '.', regex=False)
df['Último'] = pd.to_numeric(df['Último'])

# Crear variables 'lagged' (retrasadas) para la predicción
# Esto es esencial en series de tiempo para usar precios anteriores como variables predictoras
df['Último_lag1'] = df['Último'].shift(1)
df['Último_lag2'] = df['Último'].shift(2)
df['Último_lag3'] = df['Último'].shift(3)

# Eliminar las filas con valores nulos que resultaron del 'shift'
df.dropna(inplace=True)

# 3. Definir las variables predictoras (X) y la variable objetivo (y)
# Usamos las variables 'lagged' como variables predictoras (X)
X = df[['Último_lag1', 'Último_lag2', 'Último_lag3']]
# La variable objetivo (y) es el precio actual
y = df['Último']

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
# Se usa una división basada en el tiempo para evitar la fuga de datos
# El 80% de los datos se usa para entrenar y el 20% para probar
split_index = int(len(df) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# 5. Entrenar el modelo Random Forest
# Se inicializa y entrena el modelo de regresión
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Realizar predicciones y evaluar el modelo
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Evaluación del modelo ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
