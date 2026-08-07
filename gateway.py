import requests

class MLOpsGateway:
    """
    Consolidated API Gateway Matrix Layer.
    Manages schemas, dynamic URLs, timeouts, and route contracts.
    """
    def __init__(self, secrets_dict):
        # Bind the configurations dynamically from the frontend secrets layer
        self.services = {
            "churn": secrets_dict.get("CHURN_API_URL", "http://localhost:8000/predict"),
            "sentiment": secrets_dict.get("SENTIMENT_API_URL", "http://localhost:8000/predict"),
            "image": secrets_dict.get("IMAGE_API_URL", "http://localhost:8000/resize")
        }

    def _get_health_url(self, service_name):
        """Extracts base url dynamically to match the standard /health endpoint contract."""
        base_url = self.services[service_name].rsplit('/', 1)[0]
        return f"{base_url}/health"

    def check_cluster_health(self):
        """Aggregates all cluster health check results cleanly."""
        status_report = {}
        for name in self.services.keys():
            try:
                response = requests.get(self._get_health_url(name), timeout=5)
                status_report[name] = "Ready" if response.status_code == 200 else "Asleep"
            except Exception:
                status_report[name] = "Offline/Hibernating"
        return status_report

    def dispatch_inference(self, service_name, payload):
        """
        Consolidated route router. Standardizes requests, responses, 
        and timeout definitions across all backend microservices.
        """
        url = self.services.get(service_name)
        if not url:
            return {"error": f"Service Target '{service_name}' not configured."}
            
        try:
            # 60-second timeout safely absorbs any initial Render container wake-up cycles
            response = requests.post(url, json=payload, timeout=60)
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": f"The {service_name.upper()} service timed out. It is likely waking up from hibernation. Please try again."}
        except Exception as e:
            return {"error": f"Gateway Routing Exception: {str(e)}"}