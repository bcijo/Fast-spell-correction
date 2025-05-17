"""
Enhanced FastAPI server for the spell correction system.
Includes a sophisticated web interface, multiple model support,
and comprehensive API endpoints.
"""

import os
import time
import json
import logging
import torch
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Query, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import sys
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add parent directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SpellCorrector class
from src.infer import SpellCorrector

# Create the FastAPI app
app = FastAPI(
    title="Fast Spell Correction API",
    description="A lightweight, non-LLM API for spell correction",
    version="1.0.0"
)

# Get the directory of the current file
current_dir = Path(__file__).parent
static_dir = current_dir / "static"

# Mount the static directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up templates
templates = Jinja2Templates(directory=str(static_dir))

# Model configuration
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/best_model")
USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "true").lower() == "true"
MAX_PARALLEL_REQUESTS = int(os.environ.get("MAX_PARALLEL_REQUESTS", "10"))

# Initialize model dictionary
models = {}
active_model_name = "default"

# Request/Response models
class CorrectionRequest(BaseModel):
    text: str = Field(..., description="The text to correct")
    model_name: Optional[str] = Field("default", description="The model to use for correction")

class CorrectionResponse(BaseModel):
    original: str = Field(..., description="The original text")
    corrected: str = Field(..., description="The corrected text")
    latency_ms: float = Field(..., description="Processing time in milliseconds")
    changes: List[Dict[str, str]] = Field(..., description="List of changes made")
    model_name: str = Field(..., description="The model used for correction")

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Path to the model")
    quantized: bool = Field(..., description="Whether the model is quantized")
    is_active: bool = Field(..., description="Whether this is the active model")
    description: str = Field(..., description="Description of the model")

class BenchmarkResult(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    throughput: float = Field(..., description="Throughput in requests per second")
    memory_mb: float = Field(..., description="Memory usage in MB")

# Helper functions
def load_model(model_name: str, model_path: str, quantized: bool = False) -> None:
    """Load a model and add it to the models dictionary"""
    try:
        logger.info(f"Loading model '{model_name}' from {model_path} (quantized={quantized})...")
        models[model_name] = {
            "corrector": SpellCorrector(model_path, quantized=quantized),
            "path": model_path,
            "quantized": quantized,
            "description": f"Spell correction model loaded from {model_path}"
        }
        logger.info(f"Model '{model_name}' loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def find_changes(original: str, corrected: str) -> List[Dict[str, str]]:
    """
    Find the changes made between the original and corrected text
    """
    # Simple word-by-word comparison
    original_words = original.split()
    corrected_words = corrected.split()
    
    changes = []
    
    # Ensure we don't exceed the length of either list
    for i in range(min(len(original_words), len(corrected_words))):
        if original_words[i] != corrected_words[i]:
            changes.append({
                "original": original_words[i],
                "corrected": corrected_words[i],
                "position": i
            })
    
    return changes

# App startup event
@app.on_event("startup")
async def startup_event():
    """Load the default model on startup"""
    load_model("default", DEFAULT_MODEL_PATH, USE_QUANTIZED)
    logger.info("API is ready to serve requests")

# Endpoints
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the enhanced web interface"""
    return templates.TemplateResponse("enhanced_index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "models_loaded": len(models), 
        "active_model": active_model_name
    }

@app.post("/api/correct", response_model=CorrectionResponse)
def correct_text(request: CorrectionRequest):
    """
    Correct spelling errors in text
    """
    model_name = request.model_name or active_model_name
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    corrector = models[model_name]["corrector"]
    
    start_time = time.time()
    corrected_text, model_latency_ms = corrector.correct_text(request.text)
    end_time = time.time()
    
    total_latency_ms = (end_time - start_time) * 1000
    
    # Find the changes
    changes = find_changes(request.text, corrected_text)
    
    return {
        "original": request.text,
        "corrected": corrected_text,
        "latency_ms": total_latency_ms,
        "changes": changes,
        "model_name": model_name
    }

@app.get("/api/correct", response_model=CorrectionResponse)
def correct_text_get(
    text: str = Query(..., description="Text to correct"),
    model_name: str = Query("default", description="Model to use")
):
    """
    Correct spelling errors in text (GET endpoint)
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    corrector = models[model_name]["corrector"]
    
    start_time = time.time()
    corrected_text, model_latency_ms = corrector.correct_text(text)
    end_time = time.time()
    
    total_latency_ms = (end_time - start_time) * 1000
    
    # Find the changes
    changes = find_changes(text, corrected_text)
    
    return {
        "original": text,
        "corrected": corrected_text,
        "latency_ms": total_latency_ms,
        "changes": changes,
        "model_name": model_name
    }

@app.get("/api/models", response_model=List[ModelInfo])
def list_models():
    """List all loaded models"""
    model_list = []
    
    for name, model_data in models.items():
        model_list.append({
            "name": name,
            "path": model_data["path"],
            "quantized": model_data["quantized"],
            "is_active": name == active_model_name,
            "description": model_data["description"]
        })
    
    return model_list

@app.post("/api/models/{model_name}")
def set_active_model(model_name: str):
    """Set the active model"""
    global active_model_name
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    active_model_name = model_name
    return {"message": f"Active model set to '{model_name}'"}

@app.post("/api/models/load")
def load_model_endpoint(
    model_name: str = Query(..., description="Name for the model"),
    model_path: str = Query(..., description="Path to the model"),
    quantized: bool = Query(False, description="Whether to use quantized model")
):
    """Load a new model"""
    if model_name in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' already loaded")
    
    load_model(model_name, model_path, quantized)
    return {"message": f"Model '{model_name}' loaded successfully"}

@app.delete("/api/models/{model_name}")
def unload_model(model_name: str):
    """Unload a model"""
    global active_model_name
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if model_name == active_model_name:
        raise HTTPException(status_code=400, detail="Cannot unload the active model")
    
    # Remove model
    del models[model_name]
    return {"message": f"Model '{model_name}' unloaded"}

@app.get("/api/benchmark", response_model=List[BenchmarkResult])
def benchmark_models(text: str = Query("This is a sample text with some erorrs and mispellings", description="Text to benchmark with")):
    """Benchmark all loaded models"""
    results = []
    
    for name, model_data in models.items():
        corrector = model_data["corrector"]
        
        # Run multiple iterations for accuracy
        iterations = 10
        latencies = []
        
        start_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        
        for _ in range(iterations):
            start_time = time.time()
            corrector.correct_text(text)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        
        end_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        memory_used = end_memory - start_memory
        
        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000 / avg_latency  # requests per second
        
        results.append({
            "model_name": name,
            "avg_latency_ms": avg_latency,
            "throughput": throughput,
            "memory_mb": memory_used
        })
    
    return results

# Application entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", "8000"))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
