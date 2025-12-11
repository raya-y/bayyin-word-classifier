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


# =============================================================================
# BiLSTM Model Classes (needed for loading bilstm_arabert_bayyin.joblib)
# =============================================================================
import torch.nn as nn

class BiLSTMWithMeta(nn.Module):
    """
    BiLSTM with metadata support.
    Works with any input embedding (static or contextual).
    """
    def __init__(self, input_dim, categorical_cardinalities, num_numeric,
                 lstm_hidden=256, meta_proj_dim=128, num_classes=6, dropout=0.3,
                 use_bert=False, bert_model_name=None):
        super().__init__()

        self.use_bert = use_bert

        # Optional BERT encoder (not used for embedding-based models)
        if use_bert and bert_model_name:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(bert_model_name)
            input_dim = self.bert.config.hidden_size
        else:
            self.bert = None

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                           num_layers=1, batch_first=True, bidirectional=True)

        # Metadata embeddings for categorical features
        self.cat_names = list(categorical_cardinalities.keys())
        self.cat_embeddings = nn.ModuleDict()
        total_cat_emb_dim = 0
        for name, card in categorical_cardinalities.items():
            emb_dim = min(50, max(4, int(card**0.5)))
            self.cat_embeddings[name] = nn.Embedding(card, emb_dim)
            total_cat_emb_dim += emb_dim

        # Metadata projection
        self.meta_proj = nn.Sequential(
            nn.Linear(total_cat_emb_dim + num_numeric, meta_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + meta_proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, numeric_meta, categorical_meta, attention_mask=None):
        # Process text through BERT if available
        if self.use_bert and self.bert is not None:
            bert_out = self.bert(input_ids=x, attention_mask=attention_mask)
            x = bert_out.last_hidden_state
        else:
            # For static embeddings, add sequence dimension
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

        # BiLSTM processing
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1) if self.use_bert else lstm_out.squeeze(1)

        # Process metadata
        cat_embs = [self.cat_embeddings[name](categorical_meta[:, i])
                   for i, name in enumerate(self.cat_names)]
        cat_concat = torch.cat(cat_embs, dim=1) if cat_embs else \
                     torch.zeros(numeric_meta.size(0), 0, device=numeric_meta.device)

        meta_concat = torch.cat([numeric_meta, cat_concat], dim=1)
        meta_vec = self.meta_proj(meta_concat)

        # Combine and classify
        fused = torch.cat([pooled, meta_vec], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)


class BiLSTMWrapper:
    """
    Wrapper class for BiLSTM model that provides sklearn-compatible interface.
    Works with AraBERT embeddings (768-dim vectors).
    """
    def __init__(self, model, cat_cardinalities, num_numeric, num_classes=6, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.cat_cardinalities = cat_cardinalities
        self.num_numeric = num_numeric
        self.model = model
        self.default_numeric = np.zeros(num_numeric, dtype=np.float32)
        self.default_categorical = np.zeros(len(cat_cardinalities), dtype=np.int64)

    def predict(self, X):
        """Predict class labels (1-6) for AraBERT embeddings."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            batch_size = X.shape[0]
            numeric = torch.tensor(np.tile(self.default_numeric, (batch_size, 1)), dtype=torch.float32).to(self.device)
            categorical = torch.tensor(np.tile(self.default_categorical, (batch_size, 1)), dtype=torch.long).to(self.device)

            logits = self.model(X.to(self.device), numeric, categorical)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            return predictions + 1  # Convert 0-5 to 1-6

    def predict_proba(self, X):
        """Predict class probabilities for AraBERT embeddings."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            batch_size = X.shape[0]
            numeric = torch.tensor(np.tile(self.default_numeric, (batch_size, 1)), dtype=torch.float32).to(self.device)
            categorical = torch.tensor(np.tile(self.default_categorical, (batch_size, 1)), dtype=torch.long).to(self.device)

            logits = self.model(X.to(self.device), numeric, categorical)
            return torch.softmax(logits, dim=1).cpu().numpy()


# =============================================================================
# GNN Model Classes (needed for loading gnn_arabert_bayyin.joblib)
# =============================================================================

class GNNReadabilityGAT(nn.Module):
    """GNN model architecture using Graph Attention Networks."""
    def __init__(self, input_dim, hidden_dim=256, num_classes=6, dropout=0.3):
        super(GNNReadabilityGAT, self).__init__()
        try:
            from torch_geometric.nn import GATConv
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_dim*4, hidden_dim, heads=1, dropout=dropout)
        except ImportError:
            # Fallback if torch_geometric not available
            self.conv1 = None
            self.conv2 = None
            self.conv3 = None
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.bn2 = nn.BatchNorm1d(hidden_dim*4)

    def forward(self, x, edge_index):
        import torch.nn.functional as F
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GNNWrapper:
    """Wrapper with sklearn-compatible predict/predict_proba interface for GNN."""

    def __init__(self, model, input_dim=768, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.num_classes = 6
        self.model = model
        self.model.to(device)
        self.model.eval()

    def _create_edge_index(self, n_samples):
        """Create self-loop edges for inference"""
        edge_index = torch.tensor([
            list(range(n_samples)) + list(range(n_samples)),
            list(range(n_samples)) + list(range(n_samples))
        ], dtype=torch.long)
        return edge_index

    def predict(self, X):
        """Predict class labels (1-6) for AraBERT embeddings."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            # Only use first 768 dims if more provided
            if X.shape[1] > 768:
                X = X[:, :768]

            n_samples = X.shape[0]
            edge_index = self._create_edge_index(n_samples).to(self.device)
            X = X.to(self.device)

            logits = self.model(X, edge_index)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            return predictions + 1  # Convert 0-5 to 1-6

    def predict_proba(self, X):
        """Predict class probabilities for AraBERT embeddings."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            if X.shape[1] > 768:
                X = X[:, :768]

            n_samples = X.shape[0]
            edge_index = self._create_edge_index(n_samples).to(self.device)
            X = X.to(self.device)

            logits = self.model(X, edge_index)
            return torch.softmax(logits, dim=1).cpu().numpy()


# Make classes available under __main__ for joblib loading compatibility
# (the joblib file was saved from a notebook where classes were in __main__)
import sys
if '__main__' not in sys.modules:
    import types
    sys.modules['__main__'] = types.ModuleType('__main__')
sys.modules['__main__'].BiLSTMWrapper = BiLSTMWrapper
sys.modules['__main__'].BiLSTMWithMeta = BiLSTMWithMeta
sys.modules['__main__'].GNNReadabilityGAT = GNNReadabilityGAT
sys.modules['__main__'].GNNWrapper = GNNWrapper


# =============================================================================
# TextCNN Model Classes (needed for loading textcnn_arabert_bayyin.joblib)
# =============================================================================

class CNNReadability1D(nn.Module):
    """TextCNN model architecture for 1D convolutions over features."""
    def __init__(self, input_dim, num_classes=6, dropout=0.3):
        super(CNNReadability1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        import torch.nn.functional as F
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TextCNNWrapper:
    """Wrapper with sklearn-compatible predict/predict_proba interface for TextCNN."""

    def __init__(self, model, input_dim=772, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.num_classes = 6
        self.model = model
        self.model.to(device)
        self.model.eval()

    def _pad_input(self, X):
        """Pad 768-dim input to 772-dim by adding 4 zeros for stats features"""
        if X.shape[1] == 768:
            padding = torch.zeros(X.shape[0], 4, dtype=X.dtype, device=X.device)
            X = torch.cat([X, padding], dim=1)
        return X

    def predict(self, X):
        """Predict class labels (1-6) for AraBERT embeddings."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            X = self._pad_input(X)
            X = X.to(self.device)

            logits = self.model(X)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            return predictions + 1  # Convert 0-5 to 1-6

    def predict_proba(self, X):
        """Predict class probabilities for AraBERT embeddings."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            X = self._pad_input(X)
            X = X.to(self.device)

            logits = self.model(X)
            return torch.softmax(logits, dim=1).cpu().numpy()


sys.modules['__main__'].CNNReadability1D = CNNReadability1D
sys.modules['__main__'].TextCNNWrapper = TextCNNWrapper


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

def load_transformer_model_from_hf(repo_id: str, subfolder: Optional[str] = None, tokenizer_repo: Optional[str] = None) -> Tuple[Any, Any, Dict]:
    """
    Load a transformer model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        subfolder: Optional subfolder within the repo (for models in subfolders)
        tokenizer_repo: Optional separate repository for the tokenizer (for fine-tuned models)

    Returns:
        Tuple of (model, tokenizer, config)
    """
    try:
        logger.info(f"Loading transformer model from {repo_id}" + (f"/{subfolder}" if subfolder else ""))

        # Load tokenizer - use tokenizer_repo if provided, otherwise try from model repo
        if tokenizer_repo:
            logger.info(f"Loading tokenizer from base model: {tokenizer_repo}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                subfolder=subfolder
            )

        # Load model - use subfolder parameter if provided
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
            subfolder=subfolder
        )
        model.eval()  # Set to evaluation mode

        # Get model config
        config = model.config.to_dict() if hasattr(model.config, 'to_dict') else {}

        logger.info(f"Successfully loaded transformer model from {repo_id}" + (f"/{subfolder}" if subfolder else ""))
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Error loading transformer model from {repo_id}" + (f"/{subfolder}" if subfolder else "") + f": {str(e)}")
        raise

def detect_model_type(repo_id: str, subfolder: Optional[str] = None) -> str:
    """
    Attempt to detect model type by checking repository contents.
    
    Args:
        repo_id: Hugging Face repository ID
        subfolder: Optional subfolder within the repo
        
    Returns:
        'transformer' or 'classical'
    """
    try:
        # For transformer models, try to download from subfolder first
        if subfolder:
            try:
                cache_dir = snapshot_download(repo_id=repo_id, subfolder=subfolder)
                cache_path = Path(cache_dir)
                
                # Check for transformer model files in subfolder
                has_config = (cache_path / "config.json").exists()
                has_pytorch_model = (cache_path / "pytorch_model.bin").exists()
                has_safetensors = (cache_path / "model.safetensors").exists() or bool(list(cache_path.glob("*.safetensors")))
                
                if has_config and (has_pytorch_model or has_safetensors):
                    return 'transformer'
            except Exception as e:
                logger.warning(f"Could not detect type for {repo_id}/{subfolder}: {str(e)}")
                return 'transformer'  # Default to transformer for subfolder models
        
        # Fall back to checking main repo
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

def load_model_from_hf(repo_id: str, model_type: Optional[str] = None, file_path: Optional[str] = None, subfolder: Optional[str] = None, tokenizer_repo: Optional[str] = None) -> Tuple[str, str, Any, Optional[Any], Optional[Any], Dict]:
    """
    Load a model from Hugging Face Hub (supports both classical and transformer models).

    Args:
        repo_id: Hugging Face repository ID
        model_type: Optional model type ('classical' or 'transformer'). If None, will attempt detection.
        file_path: Optional specific file path in the repo (for classical models)
        subfolder: Optional subfolder within the repo (for transformer models in subfolders)
        tokenizer_repo: Optional separate repository for the tokenizer (for fine-tuned transformer models)

    Returns:
        Tuple of (model_id, detected_type, model, tokenizer, vectorizer, config)
    """
    if model_type is None:
        detected_type = detect_model_type(repo_id, subfolder)
    else:
        detected_type = model_type

    if detected_type == 'transformer':
        model, tokenizer, config = load_transformer_model_from_hf(repo_id, subfolder, tokenizer_repo)
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

