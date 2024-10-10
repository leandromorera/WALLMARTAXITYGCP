# Importar las librerías necesarias
'''import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, request, jsonify
import logging
scaler = MinMaxScaler()
# Configurar el registro de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Cargar el dataset
# Supongamos que tenemos un archivo CSV llamado "market_data.csv"
try:
    df = pd.read_csv('market_data.csv')
except FileNotFoundError as e:
    logging.error(f"Error al cargar el archivo: {e}")
    raise
# Separar variables dependientes e independientes
X = df.drop(columns=['target'])  # Todas las columnas excepto la 'target' (variables independientes)
y = df['target']  # Columna 'target' (variable dependiente)
# Escalar las variables para que estén entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Crear y entrenar el modelo
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Supongamos que queremos predecir un valor continuo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))
# Guardar el modelo entrenado
model.save('market_model.h5')
# Ejemplo de predicción
example_data = np.array([X_test[0]])  # Tomamos una muestra del conjunto de prueba
prediction = model.predict(example_data)
logging.info(f"Predicción de ejemplo: {prediction[0][0]:.2f}")'''

# Importar las librerías necesarias
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
import logging
'''
scaler = MinMaxScaler()
# Configurar el registro de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Cargar el dataset
# Supongamos que tenemos un archivo CSV llamado "market_data.csv"
try:
    df = pd.read_csv('market_data.csv')
except FileNotFoundError as e:
    logging.error(f"Error al cargar el archivo: {e}")
    raise
# Separar variables dependientes e independientes
X = df.drop(columns=['target'])  # Todas las columnas excepto la 'target' (variables independientes)
y = df['target']  # Columna 'target' (variable dependiente)
# Escalar las variables para que estén entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Crear y entrenar el modelo
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Supongamos que queremos predecir un valor continuo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))
# Guardar el modelo entrenado
model.save('market_model.h5')
# Ejemplo de predicción
example_data = np.array([X_test[0]])  # Tomamos una muestra del conjunto de prueba
prediction = model.predict(example_data)
logging.info(f"Predicción de ejemplo: {prediction[0][0]:.2f}")'''
# Configurar el registro de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Cargar el modelo para la API
model = tf.keras.models.load_model('market_model.h5')
# Escalar las variables para que estén entre 0 y 1
scaler = MinMaxScaler()
# Ajustar el scaler con datos ficticios para que esté preparado para escalar los datos de entrada
num_features = model.input_shape[1]
scaler.fit([[0] * num_features, [1] * num_features])
# Crear una API con Flask para recibir datos y predecir
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir los datos del usuario
        input_data = request.get_json()
        if not input_data or 'features' not in input_data:
            return jsonify({'error': 'Invalid input. Please provide data under the key "features".'}), 400
        data = np.array(input_data['features']).reshape(1, -1)
        if data.shape[1] != num_features:
            return jsonify({'error': f'Invalid input. Expected {num_features} features, but got {data.shape[1]}.'}), 400
        # Escalar los datos para que coincidan con el entrenamiento
        data_scaled = scaler.transform(data)
        # Realizar la predicción
        prediction = model.predict(data_scaled)
        response = {'prediction': float(prediction[0][0])}
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)