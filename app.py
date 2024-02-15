import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
with open('modelo_optimizado_sandro.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops - Trabajo Sandro')

# Controles de entrada para las características
st.sidebar.title('Parámetros de Entrada')
ram = st.sidebar.slider('RAM (GB)', 1, 64, 8)
screen_width = st.sidebar.slider('Ancho de Pantalla', 800, 4000, 1920)
screen_height = st.sidebar.slider('Alto de Pantalla', 600, 3000, 1080)
ssd = st.sidebar.slider('SSD (GB)', 0, 2000, 256)
ghz = st.sidebar.slider('GHz del CPU', 0.1, 5.0, 2.5)
type_gaming = st.sidebar.radio('¿Es Gaming?', ['No', 'Sí'])
type_notebook = st.sidebar.radio('¿Es Notebook?', ['No', 'Sí'])

# Convertir entradas a formato numérico
type_gaming = 1 if type_gaming == 'Sí' else 0
type_notebook = 1 if type_notebook == 'Sí' else 0

# Botón para realizar predicción
if st.sidebar.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    input_data = pd.DataFrame([[ram, screen_width, screen_height, ssd, ghz, type_gaming, type_notebook]],
                    columns=['Ram', 'screen_width', 'screen_height', 'SSD', 'GHz', 'TypeName_Gaming', 'TypeName_Notebook'])

    # Estandarización de las características
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Realizar predicción
    prediction = modelo.predict(input_scaled)

    # Mostrar predicción
    st.success(f'Precio predecido: {prediction[0]:.2f} Euros')




