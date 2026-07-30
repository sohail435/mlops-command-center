import streamlit as st
import requests
from PIL import Image
import io
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise MLOps Command Center",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar & API Configuration ---
st.sidebar.title("⚙️ Endpoint Configuration")
st.sidebar.markdown("Configure backend URLs (Render/Koyeb or local NodePort):")

DEFAULT_CHURN_URL = os.getenv("CHURN_API_URL", "http://localhost:30080/predict")
DEFAULT_CV_URL = os.getenv("CV_API_URL", "http://localhost:30090/resize")
DEFAULT_NLP_URL = os.getenv("NLP_API_URL", "http://localhost:30100/analyze")

churn_url = st.sidebar.text_input("Churn Predictor Endpoint", DEFAULT_CHURN_URL)
cv_url = st.sidebar.text_input("Image Resizer Endpoint", DEFAULT_CV_URL)
nlp_url = st.sidebar.text_input("DistilBERT NLP Endpoint", DEFAULT_NLP_URL)

st.sidebar.markdown("---")
navigation = st.sidebar.radio(
    "Select Microservice:",
    ["📊 Tabular: Churn Predictor", "🖼️ CV: Image Resizer", "💬 NLP: Sentiment Analysis"]
)

st.title("🚀 Enterprise MLOps Control Center")
st.markdown("---")

# ==========================================
# 📊 TAB 1: TABULAR ML - CHURN PREDICTOR
# ==========================================
if navigation == "📊 Tabular: Churn Predictor":
    st.header("📊 Customer Churn Prediction Microservice")
    st.write("Input customer demographics and usage metrics to estimate churn risk.")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.5)
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=786.0)
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    if st.button("Predict Churn", type="primary"):
        payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract_type
        }
        try:
            response = requests.post(churn_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Prediction Result:** {result}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to microservice at `{churn_url}`: {e}")

# ==========================================
# 🖼️ TAB 2: COMPUTER VISION - IMAGE RESIZER
# ==========================================
elif navigation == "🖼️ CV: Image Resizer":
    st.header("🖼️ Image Resizer Microservice")
    st.write("Upload an image and specify dimensions for processing.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Target Width", min_value=16, max_value=4096, value=300)
    with col2:
        height = st.number_input("Target Height", min_value=16, max_value=4096, value=300)

    if uploaded_file and st.button("Process Image", type="primary"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"width": width, "height": height}
        
        try:
            response = requests.post(cv_url, files=files, data=data, timeout=15)
            if response.status_code == 200:
                resized_image = Image.open(io.BytesIO(response.content))
                st.image(resized_image, caption=f"Resized Output ({width}x{height})")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to microservice at `{cv_url}`: {e}")

# ==========================================
# 💬 TAB 3: TRANSFORMER NLP - SENTIMENT
# ==========================================
elif navigation == "💬 NLP: Sentiment Analysis":
    st.header("💬 DistilBERT Sentiment Analysis Microservice")
    st.write("Analyze text sentiment in real time using your Transformer model.")

    text_input = st.text_area("Enter text to analyze:", "The MLOps deployment pipeline is running smoothly!")

    if st.button("Analyze Sentiment", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            payload = {"text": text_input}
            try:
                response = requests.post(nlp_url, json=payload, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    st.json(result)
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to microservice at `{nlp_url}`: {e}")