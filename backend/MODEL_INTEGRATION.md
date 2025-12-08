# Model Integration Guide

This document explains how the backend integrates with the models uploaded to [Hugging Face](https://huggingface.co/Raya-y/Bayyin_models/tree/main).

## Models Integrated

### Classical ML Models (using AraBERT embeddings)

1. **Random Forest** (`rf_best_halving_nsamples.pkl`)
2. **SVM Classifier** (`svm_arabert_model.pkl`)
3. **GNN Classifier** (`best_gnn_AraBERTEmbeddings.pt`)
4. **AraBERTv2 Classifier** (`best_model_arabert.joblib`)
5. **CAMeLBERT-mix Classifier** (`best_model_camelbert.joblib`)

### Transformer Models

1. **AraBERTv2 Transformer** (from `Arabertv2_D3Tok` subdirectory)
2. **CAMeLBERT-mix Transformer** (from `CAMeLBERT-mix_D3Tok` subdirectory)
3. **CAMeLBERT-MSA Classifier** (from `CAMeLBERT-msa_D3Tok` subdirectory)

### Placeholder Models (temporary)

- **XGBoost Classifier**: Uses Random Forest model temporarily
- **BiLSTM Model**: Uses AraBERT model temporarily
- **TextCNN Classifier**: Uses SVM model temporarily

## Key Features

### AraBERT Embeddings

All classical ML models use **AraBERT embeddings** for text preprocessing:

- Text is tokenized and embedded using `aubmindlab/bert-base-arabertv2` from Hugging Face
- Embeddings are generated using the [CLS] token representation
- This ensures consistent feature representation across all classical models

### Model Loading

The backend automatically:

1. Downloads models from Hugging Face Hub on startup
2. Detects model type (classical vs transformer)
3. Loads specific files when `file_path` is specified in config
4. Supports multiple file formats:
   - `.pkl` (pickle)
   - `.joblib` (joblib)
   - `.pt` (PyTorch)
   - `.zip` (archived models)

### Preprocessing Pipeline

1. **Language Detection**: Verifies input is Arabic
2. **Normalization**: 
   - Removes diacritics
   - Removes tatweel
   - Standardizes Alef/Ya variations
3. **Embedding/Tokenization**:
   - Classical models: AraBERT embeddings
   - Transformer models: Model-specific tokenizers

## Configuration

Models are configured in `config.yaml`:

```yaml
models:
  - repo_id: "Raya-y/Bayyin_models"
    type: "classical"
    name: "Random Forest"
    file_path: "rf_best_halving_nsamples.pkl"
```

- `repo_id`: Hugging Face repository ID
- `type`: "classical" or "transformer"
- `name`: Display name for the model
- `file_path`: Specific file to load (for classical models in a repo with multiple files)

## Adding New Models

To add a new model:

1. Upload to Hugging Face (if not already there)
2. Add entry to `config.yaml`:
   ```yaml
   - repo_id: "username/repo-name"
     type: "classical"  # or "transformer"
     name: "Model Name"
     file_path: "model_file.pkl"  # optional for classical models
   ```
3. Restart the backend server

## Model Types

### Classical ML Models

- Expect AraBERT embeddings as input (768-dimensional vectors)
- Models are loaded from pickle/joblib files
- Support scikit-learn compatible interfaces (`predict`, `predict_proba`)

### Transformer Models

- Use their own tokenizers
- Loaded from Hugging Face transformer format
- Support `AutoModelForSequenceClassification` interface

## Troubleshooting

### Model Not Loading

- Check that the `repo_id` and `file_path` are correct
- Verify the model file exists in the Hugging Face repository
- Check server logs for detailed error messages

### Embedding Issues

- AraBERT model is loaded on startup
- If embedding fails, check that `aubmindlab/bert-base-arabertv2` is accessible
- Ensure sufficient memory for loading the embedding model

### Prediction Errors

- Verify model expects the correct input format
- Check that model outputs are in range 1-6 (grade levels)
- Review server logs for specific error messages

