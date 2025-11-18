
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Predictor de Corners", page_icon="⚽", layout="wide")
st.title("⚽ Predictor de Corners Post-Minuto 70")
st.markdown("Predice si habrá 3 o más corners después del minuto 70")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('mlp_model.pkl')
        scaler_minmax = joblib.load('scaler_minmax.pkl')
        scaler_std = joblib.load('scaler_std.pkl')
        features = joblib.load('top_10_features.pkl')
        return model, scaler_minmax, scaler_std, features
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None, None, None

mlp_model, scaler_minmax, scaler_std, top_10_features = load_model()

if mlp_model is not None:
    st.sidebar.header("📊 Ingresa los datos del partido")
    input_data = {}
    
    for feature in top_10_features:
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.1, format="%.2f")
    
    if st.sidebar.button("🎯 Predecir", type="primary"):
        input_df = pd.DataFrame([input_data])
        try:
            input_normalized = scaler_minmax.transform(input_df)
            input_scaled = scaler_std.transform(input_normalized)
            prediction = mlp_model.predict(input_scaled)
            probability = mlp_model.predict_proba(input_scaled)
            
            st.header("📈 Resultados de la Predicción")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicción", "✅ 3+ Corners" if prediction[0] == 1 else "❌ Menos de 3 Corners")
            
            with col2:
                prob_positive = probability[0][1] * 100
                st.metric("Probabilidad de 3+ Corners", f"{prob_positive:.1f}%")
            
            st.progress(int(prob_positive))
            st.info(f"**Confianza del modelo:** {max(probability[0]) * 100:.1f}%")
            
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

else:
    st.error("No se pudieron cargar los componentes del modelo.")
