# backend/monitoring/metrics.py
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import logging

logger = logging.getLogger(__name__)

# Define metrics
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency',
    ['endpoint']
)

MODEL_CONFIDENCE = Histogram(
    'model_confidence_distribution',
    'Distribution of prediction confidence scores',
    buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
)

ACTIVE_MODEL_VERSION = Gauge(
    'active_model_version',
    'Currently active model version',
    ['model_name']
)

class MetricsMiddleware:
    def __init__(self):
        self.metrics_logger = logging.getLogger('metrics')
    
    async def track_request(self, request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            latency = time.time() - start_time
            
            if request.url.path == '/predict':
                PREDICTION_REQUESTS.labels(
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                PREDICTION_LATENCY.labels(
                    endpoint=request.url.path
                ).observe(latency)
            
            return response
            
        except Exception as e:
            logger.error(f"Request tracking error: {e}")
            raise
    
    def record_prediction_confidence(self, confidence: float):
        """Record prediction confidence metric"""
        MODEL_CONFIDENCE.observe(confidence)
    
    def set_model_version(self, version: str):
        """Set current model version metric"""
        ACTIVE_MODEL_VERSION.labels(model_name='autism_predictor').set(1)
        # Store version as label value

# FastAPI endpoint for metrics
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )