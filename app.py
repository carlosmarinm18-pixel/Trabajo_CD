import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Corners",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚽ Predictor de Corners</h1>', unsafe_allow_html=True)
st.markdown("### Predice si habrá 3 o más corners después del minuto 70")
st.markdown("---")

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
    with st.sidebar:
        st.markdown("### 📊 Datos del Partido")
        input_data = {}
        
        for feature in top_10_features:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                step=0.1,
                format="%.2f"
            )
        
        if st.button("🎯 **Predecir**", type="primary"):
            input_df = pd.DataFrame([input_data])
            try:
                input_normalized = scaler_minmax.transform(input_df)
                input_scaled = scaler_std.transform(input_normalized)
                prediction = mlp_model.predict(input_scaled)
                probability = mlp_model.predict_proba(input_scaled)
                
                st.header("📈 Resultados")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicción", 
                        "✅ 3+ Corners" if prediction[0] == 1 else "❌ Menos de 3"
                    )
                
                with col2:
                    prob_positive = probability[0][1] * 100
                    st.metric("Probabilidad", f"{prob_positive:.1f}%")
                    st.progress(int(prob_positive))
                
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.error("Error cargando el modelo")

# Footer CORREGIDO
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🚀 Desarrollado con Machine Learning | Modelo MLP con SMOTE"
    "</div>", 
    unsafe_allow_html=True
)