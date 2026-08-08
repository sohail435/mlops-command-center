import requests
import time

class MLOpsGateway:
    def __init__(self, override_urls=None):
        import streamlit as st
        self.services = {
            "churn": override_urls.get("churn") if override_urls else st.secrets.get("CHURN_API_URL", "http://localhost:8000/predict"),
            "sentiment": override_urls.get("sentiment") if override_urls else st.secrets.get("SENTIMENT_API_URL", "http://localhost:8000/predict"),
            "image": override_urls.get("image") if override_urls else st.secrets.get("IMAGE_API_URL", "http://localhost:8000/resize")
        }
        
    def get_health_url(self, service_name):
        base_url = self.services[service_name].rsplit('/', 1)[0]
        return f"{base_url}/health"

    def check_service_health(self, service_name):
        health_url = self.get_health_url(service_name)
        try:
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def route_predict(self, service_name, payload):
        url = self.services.get(service_name)
        if not url:
            return {"success": False, "error": f"Service '{service_name}' configuration missing."}
        
        start_time = time.perf_counter()  # Start Telemetry Timer
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
        url = self.services.get(service_name)
        if not url:
            return {"success": False, "error": f"Service '{service_name}' configuration missing."}
        
        start_time = time.perf_counter()  # Start Telemetry Timer
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