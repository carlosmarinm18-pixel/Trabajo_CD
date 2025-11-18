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
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header con diseño mejorado
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
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
    # Sidebar mejorado
    with st.sidebar:
        st.markdown("### 📊 Datos del Partido")
        st.markdown("Ingresa las estadísticas del partido:")
        
        input_data = {}
        
        # Dividir características en grupos para mejor organización
        st.markdown("#### Estadísticas Principales")
        for i, feature in enumerate(top_10_features[:5]):
            input_data[feature] = st.number_input(
                f"**{feature}**",
                value=0.0,
                step=0.1,
                format="%.2f",
                help=f"Valor para {feature}"
            )
        
        st.markdown("#### Estadísticas Secundarias")
        for feature in top_10_features[5:]:
            input_data[feature] = st.number_input(
                f"**{feature}**",
                value=0.0,
                step=0.1,
                format="%.2f",
                help=f"Valor para {feature}"
            )
        
        if st.button("🎯 **Hacer Predicción**", type="primary", use_container_width=True):
            st.session_state.make_prediction = True
    
    # Área principal de resultados
    if st.session_state.get('make_prediction', False):
        input_df = pd.DataFrame([input_data])
        try:
            input_normalized = scaler_minmax.transform(input_df)
            input_scaled = scaler_std.transform(input_normalized)
            prediction = mlp_model.predict(input_scaled)
            probability = mlp_model.predict_proba(input_scaled)
            
            # Resultados con mejor diseño
            st.markdown("## 📈 Resultados de la Predicción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if prediction[0] == 1:
                    st.success("### ✅ 3+ CORNERS")
                    st.metric("Predicción", "SÍ habrá 3+ corners")
                else:
                    st.error("### ❌ MENOS DE 3 CORNERS")
                    st.metric("Predicción", "NO habrá 3+ corners")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                prob_positive = probability[0][1] * 100
                st.metric(
                    "Probabilidad de 3+ Corners", 
                    f"{prob_positive:.1f}%",
                    delta=f"{prob_positive - 50:.1f}%" if prob_positive > 50 else None
                )
                
                # Barra de probabilidad con colores
                if prob_positive > 70:
                    color = "green"
                elif prob_positive > 30:
                    color = "orange"
                else:
                    color = "red"
                    
                st.markdown(f"**Nivel de confianza:**")
                st.progress(int(prob_positive))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Información adicional
            st.markdown("---")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("#### 📊 Detalles del Modelo")
                st.info(f"**Confianza del modelo:** {max(probability[0]) * 100:.1f}%")
                st.info(f"**Características usadas:** {len(top_10_features)}")
            
            with col_info2:
                st.markdown("#### 💡 Interpretación")
                if prob_positive > 70:
                    st.success("Alta probabilidad de corners en los últimos 20 minutos")
                elif prob_positive > 30:
                    st.warning("Probabilidad moderada de corners")
                else:
                    st.error("Baja probabilidad de corners")
            
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

else:
    st.error("No se pudieron cargar los componentes del modelo.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🚀 Desarrollado con Machine Learning | Modelo MLP con SMOTE"
    "</div>", 
    unsafe_allow_html=True
)