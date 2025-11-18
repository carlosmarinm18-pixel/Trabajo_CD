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

# CSS SEGURO usando Theme de Streamlit
st.markdown("""
    <style>
        .main-title {
            font-size: 40px !important;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            padding-bottom: 10px;
        }

        /* Tarjetas de métricas */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            text-align: center;
        }

        /* Botón primario */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Header SEGURO
st.markdown('<div class="main-title">⚽ Predictor de Corners</div>', unsafe_allow_html=True)
st.subheader("Predice si habrá 3 o más corners después del minuto 70")
st.markdown("---")

# ===========================
# Cargar modelo
# ===========================
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

# ===========================
# Sidebar
# ===========================
if mlp_model:
    with st.sidebar:
        st.markdown("### 📊 Datos del Partido")

        input_data = {}
        for feature in top_10_features:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                step=0.1,
                format="%.2f"
            )

        predict_btn = st.button("🎯 Predecir", type="primary")

    # ===========================
    # Predicción
    # ===========================
    if predict_btn:
        input_df = pd.DataFrame([input_data])

        try:
            input_norm = scaler_minmax.transform(input_df)
            input_scaled = scaler_std.transform(input_norm)

            prediction = mlp_model.predict(input_scaled)
            probability = mlp_model.predict_proba(input_scaled)[0][1] * 100

            st.header("📈 Resultados")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicción",
                    "✅ 3+ Corners" if prediction[0] == 1 else "❌ Menos de 3"
                )

            with col2:
                st.metric("Probabilidad", f"{probability:.1f}%")
                st.progress(int(probability))

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
else:
    st.error("No se pudo cargar el modelo.")

# ===========================
# Footer seguro
# ===========================
st.markdown("---")
st.caption("🚀 Desarrollado con Machine Learning | Modelo MLP con SMOTE")
