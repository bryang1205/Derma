import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier

# Cargar el modelo entrenado
@st.cache_data
def cargar_modelo():
    return pickle.load(open("modelo_dermatologia.pkl", "rb"))

modelo = cargar_modelo()

# Título de la app
st.title("🩺 Predicción de Enfermedades Dermatológicas")

st.markdown("Ingrese los síntomas clínicos para predecir la enfermedad de la piel.")

# Lista de características
caracteristicas = [
    "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
    "polygonal_papules", "follicular_papules", "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "melanin_incontinence", "eosinophils_infiltrate",
    "PNL_infiltrate", "fibrosis_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
    "focal_hypergranulosis", "disappearance_granular_layer", "vacuolisation_damage",
    "spongiosis", "saw_tooth_appearance", "follicular_horn_plug", "perifollicular_parakeratosis",
    "inflammatory_monoluclear_inflitrate", "band_like_infiltrate", "age"
]

# Crear campos para que el usuario ingrese los datos
entrada = []
for caracteristica in caracteristicas:
    valor = st.slider(f"{caracteristica}", 0.0, 10.0, step=1.0)
    entrada.append(valor)
# Cuando el usuario haga clic en el botón
if st.button("🔍 Predecir enfermedad"):
    datos_usuario = np.array(entrada).reshape(1, -1)
    prediccion = modelo.predict(datos_usuario)[0]
    
    enfermedades = {
        1: "Psoriasis",
        2: "Seborrheic Dermatitis",
        3: "Lichen Planus",
        4: "Pityriasis Rosea",
        5: "Chronic Dermatitis",
        6: "Pityriasis Rubra Pilaris"
    }

    st.success(f"✅ Enfermedad dermatológica predicha: **{enfermedades.get(int(prediccion), 'Desconocida')}**")
