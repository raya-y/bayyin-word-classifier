"""
AraBERT embedding module for text tokenization and embedding.
Uses AraBERT model from Hugging Face for consistent embeddings across all models.
"""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AraBERT tokenizer and model (loaded once)
_arabert_tokenizer: Optional[Any] = None
_arabert_model: Optional[Any] = None


def load_arabert_model(model_name: str = "aubmindlab/bert-base-arabertv2"):
    """
    Load AraBERT tokenizer and model for embeddings.
    
    Args:
        model_name: Hugging Face model name for AraBERT (default: aubmindlab/bert-base-arabertv2)
    """
    global _arabert_tokenizer, _arabert_model
    
    if _arabert_tokenizer is None or _arabert_model is None:
        logger.info(f"Loading AraBERT model: {model_name}")
        try:
            _arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _arabert_model = AutoModel.from_pretrained(model_name)
            _arabert_model.eval()
            logger.info("AraBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading AraBERT model: {str(e)}")
            raise


def get_arabert_embeddings(text: str, max_length: int = 512, return_cls: bool = True) -> torch.Tensor:
    """
    Get AraBERT embeddings for text.
    
    Args:
        text: Arabic text to embed
        max_length: Maximum sequence length
        return_cls: If True, return [CLS] token embedding; if False, return mean pooling
        
    Returns:
        Tensor of shape (embedding_dim,) or (1, embedding_dim)
    """
    global _arabert_tokenizer, _arabert_model
    
    if _arabert_tokenizer is None or _arabert_model is None:
        load_arabert_model()
    
    # Tokenize
    encoded = _arabert_tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get embeddings
    with torch.no_grad():
        outputs = _arabert_model(**encoded)
        
        if return_cls:
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)
        else:
            # Mean pooling over sequence
            attention_mask = encoded['attention_mask']
            embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            # Shape: (1, hidden_size)
    
    return embeddings.squeeze(0)  # Shape: (hidden_size,)


def get_arabert_tokenized(text: str, max_length: int = 512) -> dict:
    """
    Get AraBERT tokenized input (for models that need tokenized input).
    
    Args:
        text: Arabic text to tokenize
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    global _arabert_tokenizer
    
    if _arabert_tokenizer is None:
        load_arabert_model()
    
    encoded = _arabert_tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

