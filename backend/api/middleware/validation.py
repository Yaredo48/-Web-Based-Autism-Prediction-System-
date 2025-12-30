# backend/api/middleware/validation.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import time
import json

logger = logging.getLogger(__name__)

class ValidationMiddleware:
    async def __call__(self, request: Request, call_next):
        # Log request
        start_time = time.time()
        
        try:
            # Read request body
            body = await request.body()
            
            # Validate JSON for POST requests
            if request.method == "POST" and body:
                try:
                    json.loads(body.decode())
                except json.JSONDecodeError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid JSON"}
                    )
            
            # Process request
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Add headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Model-Version"] = "1.0.0"
            
            return response
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

# backend/api/validation/input_validator.py
from pydantic import BaseModel, validator, root_validator
import numpy as np

class AssessmentValidator:
    @staticmethod
    def validate_assessment_scores(scores: dict) -> bool:
        """Validate assessment scores"""
        required_keys = [f"A{i}" for i in range(1, 11)]
        
        # Check all keys present
        if not all(key in scores for key in required_keys):
            return False
        
        # Check values are 0 or 1
        if not all(scores[key] in [0, 1] for key in required_keys):
            return False
        
        return True
    
    @staticmethod
    def calculate_total_score(scores: dict) -> int:
        """Calculate total assessment score"""
        return sum(scores.values())
    
    @staticmethod
    def check_red_flags(scores: dict) -> List[str]:
        """Check for autism red flags"""
        red_flags = []
        
        # A1: Does your child look at you when you call his/her name?
        if scores.get("A1") == 1:
            red_flags.append("Does not respond to name")
        
        # A4: Does your child point to indicate that s/he wants something?
        if scores.get("A4") == 1:
            red_flags.append("Does not point to show interest")
        
        # A7: Does your child smile in response to your face or your smile?
        if scores.get("A7") == 1:
            red_flags.append("Does not smile back")
        
        return red_flags