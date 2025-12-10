"""
FastAPI main application.
Provides the classification endpoint and model management.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import uvicorn
from pathlib import Path

from app.preprocessing import detect_arabic, preprocess_text
from app.models import (
    model_registry, 
    load_model_from_hf, 
    predict_with_classical_model,
    predict_with_transformer_model,
    format_prediction
)
from app.config import load_config, get_model_repos
from app.embeddings import load_arabert_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bayyin Readability Classifier API",
    description="Backend API for Arabic text readability classification",
    version="1.0.0"
)

# Configure CORS for Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Arabic text to classify", min_length=1)


class PredictionResult(BaseModel):
    model: str
    prediction: str
    confidence: float


class ClassificationResponse(BaseModel):
    input_language: str
    prediction_results: List[PredictionResult]
    ensemble_decision: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load all models from Hugging Face on startup."""
    logger.info("Starting up... Loading models from Hugging Face")
    
    try:
        # Initialize AraBERT model for embeddings (used by all classical models)
        logger.info("Initializing AraBERT model for embeddings...")
        load_arabert_model()
        logger.info("AraBERT model initialized successfully")
        
        config = load_config()
        model_repos = get_model_repos(config)
        
        if not model_repos:
            logger.warning("No models configured. Please add models to config.yaml")
            return
        
        for model_config in model_repos:
            repo_id = model_config["repo_id"]
            model_type = model_config["type"]
            model_name = model_config["name"]
            file_path = model_config.get("file_path")
            subfolder = model_config.get("subfolder")  # Get subfolder if specified
            
            try:
                logger.info(f"Loading model: {model_name} from {repo_id}" + (f"/{subfolder}" if subfolder else ""))
                model_id, detected_type, model, tokenizer, vectorizer, config_dict = load_model_from_hf(
                    repo_id, model_type, file_path, subfolder  # Pass subfolder
                )
                
                # Use custom name if provided, otherwise use model_id
                final_model_id = model_name if model_name else model_id
                
                # Register the model (use detected type if not specified)
                model_registry.register_model(
                    model_id=final_model_id,
                    model=model,
                    model_type=detected_type,
                    tokenizer=tokenizer,
                    vectorizer=vectorizer,
                    config=config_dict
                )
                
                logger.info(f"Successfully loaded and registered model: {final_model_id}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name} from {repo_id}" + (f"/{subfolder}" if subfolder else "") + f": {str(e)}")
                continue
        
        loaded_models = model_registry.list_models()
        logger.info(f"Startup complete. Loaded {len(loaded_models)} models: {loaded_models}")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Bayyin Readability Classifier API",
        "version": "1.0.0",
        "loaded_models": model_registry.list_models()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_registry.list_models())
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Classify Arabic text using all registered models.
    
    Args:
        request: Classification request with Arabic text
        
    Returns:
        Classification results from all models plus ensemble decision
    """
    text = request.text.strip()
    
    # Detect if input is Arabic
    is_arabic, error_message = detect_arabic(text)
    if not is_arabic:
        raise HTTPException(
            status_code=400,
            detail=error_message or "Input text is not Arabic"
        )
    
    # Get all registered models
    model_ids = model_registry.list_models()
    if not model_ids:
        raise HTTPException(
            status_code=503,
            detail="No models are currently loaded. Please check server configuration."
        )
    
    prediction_results = []
    
    # Run prediction for each model
    for model_id in model_ids:
        try:
            model = model_registry.get_model(model_id)
            model_type = model_registry.get_model_type(model_id)
            
            if model_type == "transformer":
                tokenizer = model_registry.get_tokenizer(model_id)
                if tokenizer is None:
                    logger.warning(f"No tokenizer found for model {model_id}, skipping")
                    continue
                
                # Preprocess for transformer
                tokenized_input = preprocess_text(text, model_type="transformer", tokenizer=tokenizer)
                
                # Predict
                prediction, confidence = predict_with_transformer_model(
                    model, tokenizer, tokenized_input
                )
                
            else:  # classical model
                vectorizer = model_registry.get_vectorizer(model_id)
                
                # Preprocess for classical ML using AraBERT embeddings
                preprocessed_input = preprocess_text(text, model_type="classical", use_embeddings=True)
                
                # Predict
                prediction, confidence = predict_with_classical_model(
                    model, vectorizer, preprocessed_input
                )
            
            # Format result
            result = format_prediction(prediction, model_id)
            result["confidence"] = round(confidence, 2)
            prediction_results.append(result)
            
        except Exception as e:
            logger.error(f"Error predicting with model {model_id}: {str(e)}")
            # Continue with other models even if one fails
            continue
    
    if not prediction_results:
        raise HTTPException(
            status_code=500,
            detail="All model predictions failed. Please check server logs."
        )
    
    # Calculate ensemble decision (hard voting - most common prediction)
    if prediction_results:
        # Extract grade numbers from predictions
        grade_numbers = []
        for result in prediction_results:
            grade_str = result["prediction"]
            # Extract number from "Grade X"
            try:
                grade_num = int(grade_str.split()[-1])
                grade_numbers.append(grade_num)
            except:
                pass
        
        if grade_numbers:
            # Most common grade
            from collections import Counter
            most_common = Counter(grade_numbers).most_common(1)[0][0]
            ensemble_decision = f"Grade {most_common}"
        else:
            ensemble_decision = None
    else:
        ensemble_decision = None
    
    return ClassificationResponse(
        input_language="ar",
        prediction_results=prediction_results,
        ensemble_decision=ensemble_decision
    )


if __name__ == "__main__":
    config = load_config()
    api_config = config.get("api", {})
    uvicorn.run(
        "app.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=True
    )

