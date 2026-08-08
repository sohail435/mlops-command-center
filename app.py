import streamlit as st
import requests
from PIL import Image
import io
import time  # Essential for performance counter metrics

#===============================================================================
# SIMPLE HELPER FUNCTION FOR TELEMETRY
#===============================================================================
def render_telemetry_ui(result):
    """Renders MLOps telemetry metrics if latency data is available in the gateway response."""
    if "latency_ms" in result:
        st.markdown("---")
        latency = result["latency_ms"]
        
        # Determine performance status color boundaries
        if latency < 200:
            status_color = "🟢 Optimal"
        elif latency < 1000:
            status_color = "🟡 Acceptable"
        else:
            status_color = "🚨 Cold Start / High Load"
            
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.metric(label="Round-Trip Latency", value=f"{latency:.0f} ms")
        with t_col2:
            st.metric(label="Gateway Performance Status", value=status_color)

# ==============================================================================
# 🎛️ CENTRAL ROUTER & DEMAND-DRIVEN CONTROLLER
# ==============================================================================
class MLOpsGateway:
    def __init__(self, override_urls=None):
        # Fallback priority: Sidebar text fields -> Streamlit Secrets -> Local Dev defaults
        # Synchronizes variable names to ensure RESIZER_API_URL / IMAGE_API_URL consistency
        self.services = {
            "churn": override_urls.get("churn") if override_urls else st.secrets.get("CHURN_API_URL", "http://localhost:8000/predict"),
            "sentiment": override_urls.get("sentiment") if override_urls else st.secrets.get("SENTIMENT_API_URL", "http://localhost:8000/predict"),
            "image": override_urls.get("image") if override_urls else st.secrets.get("IMAGE_API_URL", "http://localhost:8000/resize")
        }
        
    def get_health_url(self, service_name):
        """Extracts the base URL and targets the /health endpoint."""
        base_url = self.services[service_name].rsplit('/', 1)[0]
        return f"{base_url}/health"

    def check_service_health(self, service_name):
        """Checks if a specific service is awake right now."""
        health_url = self.get_health_url(service_name)
        try:
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def route_predict(self, service_name, payload):
        """Unified inference router with standardized error management and latency tracking."""
        url = self.services.get(service_name)
        if not url:
            return {"success": False, "error": f"Service '{service_name}' configuration missing."}
        
        start_time = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=60)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "latency_ms": elapsed_ms}
            return {"success": False, "error": f"API Error ({response.status_code}): {response.text}", "latency_ms": elapsed_ms}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out. The engine is waking up from free-tier hibernation. Please try again!"}
        except Exception as e:
            return {"success": False, "error": f"Failed to connect to backend: {str(e)}"}

    def route_file_process(self, service_name, files, data):
        """Specialized routing method for multipart/form-data operations with latency tracking."""
        url = self.services.get(service_name)
        if not url:
            return {"success": False, "error": f"Service '{service_name}' configuration missing."}
        
        start_time = time.perf_counter()
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                return {"success": True, "content": response.content, "latency_ms": elapsed_ms}
            return {"success": False, "error": f"API Error ({response.status_code}): {response.text}", "latency_ms": elapsed_ms}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Image processor timed out during initialization. Please try again."}
        except Exception as e:
            return {"success": False, "error": f"File routing failure: {str(e)}"}

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise MLOps Command Center",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar & API Configuration ---
st.sidebar.title("🔋 System Status Command Center")

# Step 1: Sidebar Setup Inputs
st.sidebar.markdown("Configure backend URLs (Render/Koyeb or local NodePort):")
DEFAULT_CHURN_URL = st.secrets.get("CHURN_API_URL", "http://localhost:30080/predict")
DEFAULT_NLP_URL = st.secrets.get("SENTIMENT_API_URL", "http://localhost:30100/predict")
DEFAULT_CV_URL = st.secrets.get("IMAGE_API_URL", st.secrets.get("RESIZER_API_URL", "http://localhost:30090/resize"))

churn_url_input = st.sidebar.text_input("Churn Predictor Endpoint", DEFAULT_CHURN_URL)
cv_url_input = st.sidebar.text_input("Image Resizer Endpoint", DEFAULT_CV_URL)
nlp_url_input = st.sidebar.text_input("DistilBERT NLP Endpoint", DEFAULT_NLP_URL)

# Step 2: Initialize or override gateway state based on UI values
override_map = {
    "churn": churn_url_input,
    "sentiment": nlp_url_input,
    "image": cv_url_input
}
gateway = MLOpsGateway(override_urls=override_map)

# Health Check Trigger Matrix
if st.sidebar.button("Check Backend Status"):
    for service in ["churn", "sentiment", "image"]:
        is_alive = gateway.check_service_health(service)
        if is_alive:
            st.sidebar.success(f"🟢 {service.upper()} Service: Ready")
        else:
            st.sidebar.warning(f"🟡 {service.upper()} Service: Asleep (Will trigger cold start)")

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
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=786.0)
        contract_type = st.selectbox("Contract Type", ["month-to-month", "one year", "two year"])
        plan = st.selectbox("Plan Type", ["basic", "premium"])

    if st.button("Predict Churn", type="primary"):
        payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract_type,
            "age": age,
            "plan": plan
        }
        
        with st.spinner("Routing request through MLOpsGateway..."):
            result = gateway.route_predict("churn", payload)
            
        if result.get("success"):
            data = result['data']
            
            # Robust extraction layer covering absolute keys and fallback keys across environments
            is_churn = data.get("churn_prediction") if data.get("churn_prediction") is not None else (data.get("churn") or data.get("prediction"))
            prob = data.get("churn_probability") if data.get("churn_probability") is not None else (data.get("probability") or data.get("confidence"))
            
            st.markdown("### 🎯 Analysis Result")
            
            if is_churn == 1 or is_churn is True:
                st.error("🚨 **High Risk:** This customer is highly likely to churn.")
            else:
                st.success("✅ **Low Risk:** This customer is likely to stay retained.")
                
            # Render the probability cleanly as a metric card
            if prob is not None:
                st.metric(label="Churn Probability", value=f"{float(prob) * 100:.1f}%")
                
            # Render Telemetry UI component
            render_telemetry_ui(result)
        else:
            st.error(result.get("error"))

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
        
        with st.spinner("Uploading and resizing image via Gateway matrix..."):
            result = gateway.route_file_process("image", files=files, data=data)
            
        if result.get("success"):
            # Get raw image data bytes
            image_bytes = result["content"]
            
            # Display image in UI
            resized_image = Image.open(io.BytesIO(image_bytes))
            st.image(resized_image, caption=f"Resized Output ({width}x{height})")
            
            # Add space and the clean download button
            st.markdown("### 📥 Download Processed Image")
            st.download_button(
                label="Download Resized Image",
                data=image_bytes,
                file_name=f"resized_{width}x{height}_{uploaded_file.name}",
                mime=uploaded_file.type
            )
            
            # Render Telemetry UI component
            render_telemetry_ui(result)
        else:
            st.error(result.get("error"))

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
            
            with st.spinner("Evaluating payload context..."):
                result = gateway.route_predict("sentiment", payload)
                
            if result.get("success"):
                data = result["data"]
                
                # Extract fields based on your DistilBERT pipeline output
                label = data.get("label") or data.get("sentiment")
                score = data.get("score") or data.get("confidence")
                
                st.markdown("### 📊 Model Inference Result")
                
                # Visual display based on label value
                if label and "POS" in str(label).upper():
                    st.success(f"😊 **Positive Sentiment Detected**")
                elif label and "NEG" in str(label).upper():
                    st.error(f"😢 **Negative Sentiment Detected**")
                else:
                    st.info(f"😐 **Neutral Sentiment Detected**")
                
                if score is not None:
                    st.metric(label="Model Confidence Score", value=f"{float(score) * 100:.2f}%")
                
                # Render Telemetry UI component
                render_telemetry_ui(result)
            else:
                st.error(result.get("error"))