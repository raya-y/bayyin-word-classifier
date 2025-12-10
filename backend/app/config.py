"""
Configuration management module.
Handles loading model configurations from config file.
"""
import yaml
import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in backend directory.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config.yaml in backend directory
        backend_dir = Path(__file__).parent.parent
        config_path = backend_dir / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {
            "models": [],
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Set defaults
        if "models" not in config:
            config["models"] = []
        if "api" not in config:
            config["api"] = {"host": "0.0.0.0", "port": 8000}
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise


def get_model_repos(config: Dict) -> List[Dict]:
    """
    Extract model repository configurations from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of model repository configurations
    """
    models = config.get("models", [])
    
    # Support both simple list of strings and list of dicts
    model_repos = []
    for model in models:
        if isinstance(model, str):
            # Simple format: just repo ID
            model_repos.append({
                "repo_id": model, 
                "type": None, 
                "name": model.split("/")[-1],
                "file_path": None,
                "subfolder": None
            })
        elif isinstance(model, dict):
            # Extended format: dict with repo_id, type, name, file_path, subfolder
            repo_id = model.get("repo_id") or model.get("repo")
            if not repo_id:
                logger.warning(f"Invalid model config: {model}, skipping")
                continue
            model_repos.append({
                "repo_id": repo_id,
                "type": model.get("type"),
                "name": model.get("name", repo_id.split("/")[-1]),
                "file_path": model.get("file_path"),  # Optional specific file path
                "subfolder": model.get("subfolder")   # Optional subfolder for transformers
            })
    
    return model_repos