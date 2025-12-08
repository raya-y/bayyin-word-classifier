"""
Arabic text preprocessing module.
Handles language detection, normalization, and tokenization for Arabic text.
Uses AraBERT for embeddings.
"""
import re
import unicodedata
from typing import Tuple, Optional, Union
import torch
import langdetect
from langdetect import DetectorFactory
from app.embeddings import get_arabert_embeddings, get_arabert_tokenized, load_arabert_model

# Set seed for consistent language detection
DetectorFactory.seed = 0


def detect_arabic(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if the input text is Arabic.
    Uses character-based detection first (more reliable), then langdetect as secondary check.
    
    Args:
        text: Input text to check
        
    Returns:
        Tuple of (is_arabic: bool, error_message: Optional[str])
    """
    if not text or not text.strip():
        return False, "Input text is empty"
    
    # First, check if text contains Arabic characters (most reliable method)
    # Arabic Unicode ranges
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    has_arabic_chars = arabic_pattern.search(text)
    
    # Count Arabic characters vs total characters to determine if it's primarily Arabic
    arabic_chars = arabic_pattern.findall(text)
    total_chars = len([c for c in text if c.strip()])  # Non-whitespace characters
    
    # If text has Arabic characters and they make up a significant portion, accept it
    if has_arabic_chars and total_chars > 0:
        arabic_ratio = len(arabic_chars) / total_chars if total_chars > 0 else 0
        # Accept if at least 30% of characters are Arabic (handles mixed text)
        if arabic_ratio >= 0.3:
            return True, None
    
    # Secondary check: use langdetect (but don't rely solely on it due to Arabic/Farsi confusion)
    try:
        detected_lang = langdetect.detect(text)
        # Accept if langdetect says Arabic OR if we have Arabic characters
        if detected_lang == 'ar' or has_arabic_chars:
            return True, None
        else:
            # If langdetect says something else but we have Arabic chars, still accept
            if has_arabic_chars:
                return True, None
            return False, f"Input text is not Arabic. Detected language: {detected_lang}. Please enter Arabic text for classification."
    except Exception as e:
        # If langdetect fails, fall back to character check
        if has_arabic_chars:
            return True, None
        return False, "Could not detect Arabic text. Please ensure the input contains Arabic characters."


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text by:
    - Removing diacritics (tashkeel)
    - Removing tatweel (elongation character)
    - Standardizing Alef variations
    - Standardizing Ya variations
    
    Args:
        text: Arabic text to normalize
        
    Returns:
        Normalized Arabic text
    """
    # Remove diacritics (Arabic diacritical marks)
    # Range: \u064B-\u065F, \u0670, \u06D6-\u06ED
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
    
    # Remove tatweel (elongation character)
    text = text.replace('\u0640', '')
    
    # Standardize Alef variations to standard Alef (ا)
    # Alef variations: \u0622 (آ), \u0623 (أ), \u0625 (إ), \u0627 (ا)
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    
    # Standardize Ya variations to standard Ya (ي)
    # Ya variations: \u0649 (ى), \u064A (ي)
    text = text.replace('\u0649', '\u064A')
    
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def preprocess_for_classical_ml(text: str, use_embeddings: bool = True) -> Union[str, 'torch.Tensor']:
    """
    Preprocess text for classical ML models (RandomForest, SVM, etc.).
    Uses AraBERT embeddings by default, or normalized text if embeddings disabled.
    
    Args:
        text: Arabic text to preprocess
        use_embeddings: If True, return AraBERT embeddings; if False, return normalized text
        
    Returns:
        AraBERT embeddings tensor or normalized text string
    """
    normalized = normalize_arabic(text)
    
    if use_embeddings:
        # Ensure AraBERT is loaded
        load_arabert_model()
        # Get embeddings
        embeddings = get_arabert_embeddings(normalized, return_cls=True)
        return embeddings
    else:
        return normalized


def preprocess_for_transformer(text: str, tokenizer) -> dict:
    """
    Preprocess text for transformer models using the provided tokenizer.
    
    Args:
        text: Arabic text to preprocess
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Tokenized input dictionary ready for model inference
    """
    normalized = normalize_arabic(text)
    
    # Use the tokenizer to encode the text
    # Most transformers expect max_length, padding, truncation, return_tensors
    encoded = tokenizer(
        normalized,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'  # Return PyTorch tensors
    )
    
    # Convert to format expected by models (remove batch dimension if needed)
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }


def preprocess_text(text: str, model_type: str = 'classical', tokenizer=None, use_embeddings: bool = True) -> Union[str, dict, 'torch.Tensor']:
    """
    Main preprocessing function that routes to appropriate preprocessing based on model type.
    
    Args:
        text: Arabic text to preprocess
        model_type: Type of model ('classical' or 'transformer')
        tokenizer: Optional tokenizer for transformer models
        use_embeddings: For classical models, whether to use AraBERT embeddings (default: True)
        
    Returns:
        Preprocessed text (tensor for classical with embeddings, dict for transformer, string for classical without embeddings)
    """
    if model_type == 'transformer' and tokenizer is not None:
        return preprocess_for_transformer(text, tokenizer)
    else:
        return preprocess_for_classical_ml(text, use_embeddings=use_embeddings)

