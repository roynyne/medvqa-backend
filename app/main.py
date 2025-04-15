# app/main.py

from fastapi import FastAPI
from app.api import router as api_router

app = FastAPI(
    title="MedVQA API",
    description="Backend for ResNet-18 + PubMedBERT MedVQA model",
    version="1.0"
)

# Add a root check endpoint
@app.get("/")
def root():
    return {"message": "MedVQA API is running"}

# Include actual prediction routes
app.include_router(api_router)
