"""
A simple FastAPI application for spell correction.
"""

import os
import time
import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
import torch
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SpellCorrector class from infer.py
from src.infer import SpellCorrector

app = FastAPI(
    title="Fast Spell Correction API",
    description="A lightweight API for spell correction using a quantized masked language model",
    version="1.0.0"
)

# Initialize the SpellCorrector
model_path = os.environ.get("MODEL_PATH", "./models/best_model")
quantized = os.environ.get("USE_QUANTIZED", "true").lower() == "true"

corrector = None

class CorrectionRequest(BaseModel):
    text: str

class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    latency_ms: float

@app.on_event("startup")
async def startup_event():
    global corrector
    print(f"Loading model from {model_path} (quantized={quantized})...")
    corrector = SpellCorrector(model_path, quantized=quantized)
    print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Fast Spell Correction API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": corrector is not None}

@app.post("/correct", response_model=CorrectionResponse)
def correct_text(request: CorrectionRequest):
    if corrector is None:
        return {"error": "Model not loaded yet"}
    
    start_time = time.time()
    corrected_text, model_latency_ms = corrector.correct_text(request.text)
    end_time = time.time()
    
    total_latency_ms = (end_time - start_time) * 1000
    
    return {
        "original": request.text,
        "corrected": corrected_text,
        "latency_ms": total_latency_ms
    }

@app.get("/correct", response_model=CorrectionResponse)
def correct_text_get(text: str = Query(..., description="Text to correct")):
    if corrector is None:
        return {"error": "Model not loaded yet"}
    
    start_time = time.time()
    corrected_text, model_latency_ms = corrector.correct_text(text)
    end_time = time.time()
    
    total_latency_ms = (end_time - start_time) * 1000
    
    return {
        "original": text,
        "corrected": corrected_text,
        "latency_ms": total_latency_ms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
