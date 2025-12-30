
# Generate API documentation
# backend/docs.py
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Autism Prediction System API",
        version="1.0.0",
        description="""
        # Autism Prediction System API
        
        ## Overview
        This API provides autism prediction based on behavioral assessment scores.
        
        ## Important Notes
        - This is a screening tool, not a diagnostic tool
        - Always consult with healthcare professionals
        - Prediction confidence below 60% should be treated with caution
        
        ## Authentication
        Currently open for development. Will implement JWT tokens in production.
        
        ## Rate Limiting
        100 requests per minute per IP address.
        
        ## Error Codes
        - 400: Bad Request
        - 422: Validation Error
        - 429: Too Many Requests
        - 500: Internal Server Error
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi