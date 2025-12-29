from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.config import settings
import datetime
#from datetime import datetime

app=FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs",if settings.DEBUG else None,
    redoc_url="/redoc",if settings.DEBUG else None,
    #openapi_url="/openapi.json",if settings.DEBUG else None,
    #description=settings.DESCRIPTION   
    
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Autism Prediction System API",
        "version": settings.VERSION,
        "docs": "/docs" #if settings.DEBUG else "Disabled in production"
    }
    

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )