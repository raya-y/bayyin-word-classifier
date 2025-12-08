"""
Model loading and inference module.
Handles loading models from Hugging Face Hub and running predictions.
"""
import os
import pickle
import zipfile
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import torch
import numpy as np
import joblib
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing loaded models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.model_types: Dict[str, str] = {}  # 'classical' or 'transformer'
        self.model_configs: Dict[str, Dict] = {}
    
    def register_model(self, model_id: str, model: Any, model_type: str, 
                      tokenizer: Optional[Any] = None, 
                      vectorizer: Optional[Any] = None,
                      config: Optional[Dict] = None):
        """Register a loaded model in the registry."""
        self.models[model_id] = model
        self.model_types[model_id] = model_type
        if tokenizer:
            self.tokenizers[model_id] = tokenizer
        if vectorizer:
            self.vectorizers[model_id] = vectorizer
        if config:
            self.model_configs[model_id] = config
        logger.info(f"Registered model: {model_id} (type: {model_type})")
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a registered model."""
        return self.models.get(model_id)
    
    def get_tokenizer(self, model_id: str) -> Optional[Any]:
        """Get a registered tokenizer."""
        return self.tokenizers.get(model_id)
    
    def get_vectorizer(self, model_id: str) -> Optional[Any]:
        """Get a registered vectorizer."""
        return self.vectorizers.get(model_id)
    
    def get_model_type(self, model_id: str) -> Optional[str]:
        """Get the type of a registered model."""
        return self.model_types.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())


# Global model registry
model_registry = ModelRegistry()


def load_classical_model_from_hf(repo_id: str, file_path: Optional[str] = None) -> Tuple[Any, Optional[Any], Dict]:
    """
    Load a classical ML model (pickle/joblib/zip) from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        file_path: Optional specific file path in the repo (e.g., "rf_best_halving_nsamples.pkl")
        
    Returns:
        Tuple of (model, vectorizer, config)
    """
    try:
        logger.info(f"Loading classical model from {repo_id}, file: {file_path}")
        
        model = None
        vectorizer = None
        config = {}
        
        if file_path:
            # Download specific file
            local_path = hf_hub_download(repo_id=repo_id, filename=file_path)
            cache_path = Path(local_path).parent
            file_to_load = Path(local_path)
        else:
            # Download the entire repository
            cache_dir = snapshot_download(repo_id=repo_id)
            cache_path = Path(cache_dir)
            file_to_load = None
        
        # Load specific file if provided
        if file_to_load and file_to_load.exists():
            logger.info(f"Loading from specific file: {file_to_load.name}")
            if file_to_load.suffix == '.joblib':
                obj = joblib.load(file_to_load)
            elif file_to_load.suffix == '.pt':
                # PyTorch model files require torch.load()
                obj = torch.load(file_to_load, map_location='cpu')
            elif file_to_load.suffix == '.pkl':
                with open(file_to_load, 'rb') as f:
                    obj = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file type: {file_to_load.suffix}")
            
            if hasattr(obj, 'predict') or hasattr(obj, 'predict_proba'):
                model = obj
                logger.info(f"Loaded model from {file_to_load.name}")
            elif hasattr(obj, 'transform') or hasattr(obj, 'fit_transform'):
                vectorizer = obj
                logger.info(f"Loaded vectorizer from {file_to_load.name}")
        
        # If no specific file or model not found, search repository
        if model is None:
            # Check for zip file
            zip_files = list(cache_path.glob("*.zip"))
            if zip_files:
                logger.info(f"Found zip file: {zip_files[0]}")
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_ref.extractall(temp_dir)
                        temp_path = Path(temp_dir)
                        
                        # Look for pickle, joblib, and PyTorch files
                        for ext in ['*.pkl', '*.joblib', '*.pt']:
                            files = list(temp_path.rglob(ext))
                            for model_file in files:
                                try:
                                    if ext == '*.joblib':
                                        obj = joblib.load(model_file)
                                    elif ext == '*.pt':
                                        # PyTorch model files require torch.load()
                                        obj = torch.load(model_file, map_location='cpu')
                                    else:
                                        with open(model_file, 'rb') as f:
                                            obj = pickle.load(f)
                                    
                                    if hasattr(obj, 'predict') or hasattr(obj, 'predict_proba'):
                                        if model is None:
                                            model = obj
                                            logger.info(f"Loaded model from {model_file.name}")
                                    elif hasattr(obj, 'transform') or hasattr(obj, 'fit_transform'):
                                        if vectorizer is None:
                                            vectorizer = obj
                                            logger.info(f"Loaded vectorizer from {model_file.name}")
                                except Exception as e:
                                    logger.warning(f"Error loading {model_file.name}: {str(e)}")
                                    continue
            
            # Check for direct pickle, joblib, and PyTorch files
            for ext in ['*.pkl', '*.joblib', '*.pt']:
                files = list(cache_path.rglob(ext))
                for model_file in files:
                    try:
                        if ext == '*.joblib':
                            obj = joblib.load(model_file)
                        elif ext == '*.pt':
                            # PyTorch model files require torch.load()
                            obj = torch.load(model_file, map_location='cpu')
                        else:
                            with open(model_file, 'rb') as f:
                                obj = pickle.load(f)
                        
                        if hasattr(obj, 'predict') or hasattr(obj, 'predict_proba'):
                            if model is None:
                                model = obj
                                logger.info(f"Loaded model from {model_file.name}")
                        elif hasattr(obj, 'transform') or hasattr(obj, 'fit_transform'):
                            if vectorizer is None:
                                vectorizer = obj
                                logger.info(f"Loaded vectorizer from {model_file.name}")
                    except Exception as e:
                        logger.warning(f"Error loading {model_file.name}: {str(e)}")
                        continue
        
        # Check for config.json
        config_file = cache_path / "config.json"
        if config_file.exists():
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        if model is None:
            raise ValueError(f"Could not find a valid model in {repo_id}")
        
        logger.info(f"Successfully loaded classical model from {repo_id}")
        return model, vectorizer, config
        
    except Exception as e:
        logger.error(f"Error loading classical model from {repo_id}: {str(e)}")
        raise


def load_transformer_model_from_hf(repo_id: str) -> Tuple[Any, Any, Dict]:
    """
    Load a transformer model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    try:
        logger.info(f"Loading transformer model from {repo_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        model.eval()  # Set to evaluation mode
        
        # Get model config
        config = model.config.to_dict() if hasattr(model.config, 'to_dict') else {}
        
        logger.info(f"Successfully loaded transformer model from {repo_id}")
        return model, tokenizer, config
        
    except Exception as e:
        logger.error(f"Error loading transformer model from {repo_id}: {str(e)}")
        raise


def detect_model_type(repo_id: str) -> str:
    """
    Attempt to detect model type by checking repository contents.
    
    Args:
        repo_id: Hugging Face repository ID
        
    Returns:
        'transformer' or 'classical'
    """
    try:
        cache_dir = snapshot_download(repo_id=repo_id)
        cache_path = Path(cache_dir)
        
        # Check for transformer model files
        has_config = (cache_path / "config.json").exists()
        has_pytorch_model = (cache_path / "pytorch_model.bin").exists()
        has_safetensors = (cache_path / "model.safetensors").exists() or bool(list(cache_path.glob("*.safetensors")))
        
        if has_config and (has_pytorch_model or has_safetensors):
            return 'transformer'
        
        # Check for classical ML files
        if list(cache_path.rglob("*.pkl")) or list(cache_path.rglob("*.zip")):
            return 'classical'
        
        # Default to transformer if we can't determine
        logger.warning(f"Could not determine model type for {repo_id}, defaulting to transformer")
        return 'transformer'
        
    except Exception as e:
        logger.warning(f"Error detecting model type for {repo_id}: {str(e)}, defaulting to transformer")
        return 'transformer'


def load_model_from_hf(repo_id: str, model_type: Optional[str] = None, file_path: Optional[str] = None) -> Tuple[str, str, Any, Optional[Any], Optional[Any], Dict]:
    """
    Load a model from Hugging Face Hub (supports both classical and transformer models).
    
    Args:
        repo_id: Hugging Face repository ID
        model_type: Optional model type ('classical' or 'transformer'). If None, will attempt detection.
        file_path: Optional specific file path in the repo (for classical models)
        
    Returns:
        Tuple of (model_id, detected_type, model, tokenizer, vectorizer, config)
    """
    if model_type is None:
        detected_type = detect_model_type(repo_id)
    else:
        detected_type = model_type
    
    if detected_type == 'transformer':
        model, tokenizer, config = load_transformer_model_from_hf(repo_id)
        return repo_id, detected_type, model, tokenizer, None, config
    else:
        model, vectorizer, config = load_classical_model_from_hf(repo_id, file_path)
        return repo_id, detected_type, model, None, vectorizer, config


def predict_with_classical_model(model: Any, vectorizer: Optional[Any], 
                                 preprocessed_input: Union[str, torch.Tensor]) -> Tuple[int, float]:
    """
    Run prediction with a classical ML model.
    
    Args:
        model: The trained model
        vectorizer: Optional vectorizer for feature extraction
        preprocessed_input: Preprocessed text (string) or AraBERT embeddings (tensor)
        
    Returns:
        Tuple of (prediction, confidence)
    """
    try:
        # Handle tensor embeddings
        if isinstance(preprocessed_input, torch.Tensor):
            # Convert tensor to numpy array
            embeddings_np = preprocessed_input.cpu().numpy()
            # Reshape if needed (should be 1D, make it 2D for sklearn)
            if len(embeddings_np.shape) == 1:
                embeddings_np = embeddings_np.reshape(1, -1)
            text_vector = embeddings_np
        else:
            # Vectorize the text if vectorizer is provided
            if vectorizer:
                text_vector = vectorizer.transform([preprocessed_input])
            else:
                # If no vectorizer, assume model expects raw text or has built-in vectorization
                text_vector = [preprocessed_input]
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            prediction = model.predict(text_vector)[0]
            confidence = float(max(probabilities))
        else:
            prediction = model.predict(text_vector)[0]
            confidence = 1.0  # Default confidence if model doesn't provide probabilities
        
        # Ensure prediction is an integer (grade level 1-6)
        if isinstance(prediction, (list, np.ndarray)):
            prediction = int(prediction[0])
        else:
            prediction = int(prediction)
        
        # Map to grade level if needed (some models might output 0-5 instead of 1-6)
        if prediction < 1:
            prediction = 1
        elif prediction > 6:
            prediction = 6
        
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"Error in classical model prediction: {str(e)}")
        raise


def predict_with_transformer_model(model: Any, tokenizer: Any, 
                                  tokenized_input: dict) -> Tuple[int, float]:
    """
    Run prediction with a transformer model.
    
    Args:
        model: The trained transformer model
        tokenizer: The tokenizer
        tokenized_input: Tokenized input dictionary
        
    Returns:
        Tuple of (prediction, confidence)
    """
    try:
        with torch.no_grad():
            # Move inputs to same device as model
            device = next(model.parameters()).device
            input_ids = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)
            
            # Run inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][predicted_class].item())
            
            # Map to grade level (1-6)
            # Models typically output 0-5 (0-indexed) or 1-6 (1-indexed)
            # Apply consistent transformation: always add 1 (assuming 0-indexed), then clamp to 1-6
            predicted_class_int = int(predicted_class)
            
            # Apply consistent offset: add 1 to convert from 0-indexed to 1-indexed
            # This handles both cases: 0-5 maps to 1-6, and 6+ gets clamped appropriately
            prediction = predicted_class_int + 1
            
            # Clamp to valid grade range (1-6)
            prediction = max(1, min(prediction, 6))
            
            return prediction, confidence
            
    except Exception as e:
        logger.error(f"Error in transformer model prediction: {str(e)}")
        raise


def format_prediction(prediction: int, model_name: str) -> Dict[str, Any]:
    """
    Format prediction result for API response.
    
    Args:
        prediction: Predicted grade level (1-6)
        model_name: Name of the model
        
    Returns:
        Formatted prediction dictionary
    """
    grade_map = {
        1: "Grade 1",
        2: "Grade 2",
        3: "Grade 3",
        4: "Grade 4",
        5: "Grade 5",
        6: "Grade 6"
    }
    
    return {
        "model": model_name,
        "prediction": grade_map.get(prediction, f"Grade {prediction}"),
        "confidence": 0.0  # Will be set by caller
    }

