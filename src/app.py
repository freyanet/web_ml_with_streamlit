import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Predictor de Diabetes", 
    page_icon="🩺", 
    layout="centered"
)


st.title("🩺 Asistente de Predicción de Diabetes")
st.write(
    "Esta aplicación utiliza un modelo de Árbol de Decisión para evaluar el riesgo "
    "de diabetes basándose en datos clínicos."
)

@st.cache_resource
def cargar_modelo():
    ruta_src = os.path.dirname(__file__)
    
    ruta_raiz = os.path.abspath(os.path.join(ruta_src, os.pardir))
    

    ruta_modelo = os.path.join(ruta_raiz, "models", "decision_tree_regressor_default_42.sav")
    
    with open(ruta_modelo, "rb") as archivo:
        modelo = pickle.load(archivo)
    return modelo

modelo = cargar_modelo()


st.header("📊 Datos Clínicos del Paciente")


col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Embarazos (Pregnancies):", min_value=0, max_value=20, value=0, step=1)
    glucose = st.slider("Glucosa en sangre (Glucose):", min_value=0, max_value=200, value=120)
    blood_pressure = st.slider("Presión Arterial Diastólica (BloodPressure):", min_value=0, max_value=130, value=70)
    skin_thickness = st.slider("Espesor del pliegue cutáneo (SkinThickness):", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.slider("Insulina sérica (Insulin):", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Índice de Masa Corporal (BMI):", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    diabetes_pedigree = st.number_input("Función del Pedigrí de Diabetes (DiabetesPedigreeFunction):", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.slider("Edad (Age):", min_value=1, max_value=120, value=33)

st.write("---")


if st.button("🚀 Evaluar Paciente"):
    
    
    datos_paciente = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Realizar la predicción
    prediccion = modelo.predict(datos_paciente)
    
    st.subheader("📋 Resultado del Análisis")
    
    
    if prediccion[0] >= 0.5:
        st.error(f"⚠️ **Riesgo Alto:** El modelo predice un resultado POSITIVO para diabetes (Valor: {prediccion[0]:.2f}).")
    else:
        st.success(f"✅ **Riesgo Bajo:** El modelo predice un resultado NEGATIVO para diabetes (Valor: {prediccion[0]:.2f}).")