# Bayyin Readability Classifier Backend

Backend service for the Bayyin Arabic text readability classifier. This service loads pretrained models from Hugging Face Hub and provides a REST API for text classification.

## Features

- **Automatic Model Loading**: Loads models directly from Hugging Face Hub using the HF API
- **Multiple Model Types**: Supports both classical ML models (Pickle/zip) and transformer models
- **Arabic Text Processing**: Comprehensive preprocessing including language detection, normalization, and tokenization
- **Ensemble Predictions**: Runs all registered models and provides ensemble decision
- **FastAPI**: Modern, fast web framework with automatic API documentation

## Setup

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Installation

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure models**:
   Edit `config.yaml` and add your Hugging Face model repository IDs:
   ```yaml
   models:
     - "your-username/your-model-repo"
     - repo_id: "your-username/another-model"
       type: "transformer"  # optional
       name: "custom-name"  # optional
   ```

3. **Set Hugging Face token (if needed)**:
   If your models are private, set your HF token:
   ```bash
   export HF_TOKEN=your_token_here
   ```
   Or create a `.env` file (not included in repo for security).

## Running the Service

### Development Mode

```bash
cd backend
python -m app.main
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

Build the image:
```bash
docker build -t bayyin-backend .
```

Run the container:
```bash
docker run -p 8000:8000 bayyin-backend
```

## How Models are Loaded from Hugging Face

The backend automatically loads models from Hugging Face Hub at startup:

1. **Model Detection**: The system attempts to detect model type by checking repository contents:
   - **Transformer models**: Identified by presence of `config.json` and model weights (`.bin`, `.safetensors`)
   - **Classical ML models**: Identified by presence of `.pkl` or `.zip` files

2. **Loading Process**:
   - **Transformer models**: Uses `transformers` library to load model and tokenizer
   - **Classical ML models**: Downloads repository, extracts pickle/zip files, loads model and vectorizer (if present)

3. **Caching**: Models are cached in memory after loading for fast inference

4. **Configuration**: Model repository IDs are specified in `config.yaml`. You can specify:
   - Simple format: `"username/repo-name"`
   - Extended format with explicit type and custom name

## API Usage

### Endpoint: `POST /classify`

Classify Arabic text using all registered models.

**Request**:
```json
{
  "text": "النص العربي المراد تصنيفه"
}
```

**Response**:
```json
{
  "input_language": "ar",
  "prediction_results": [
    {
      "model": "rf_v1",
      "prediction": "Grade 5",
      "confidence": 0.82
    },
    {
      "model": "svm_v1",
      "prediction": "Grade 6",
      "confidence": 0.64
    },
    {
      "model": "arabert_v1",
      "prediction": "Grade 5",
      "confidence": 0.91
    }
  ],
  "ensemble_decision": "Grade 5"
}
```

### Example cURL

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "النص العربي المراد تصنيفه"}'
```

### Example JavaScript (for Lovable frontend)

```javascript
const classifyText = async (text) => {
  try {
    const response = await fetch('http://localhost:8000/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error('Classification failed');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};
```

### Frontend Integration

Update your frontend `classifyWord` function in `src/pages/Index.tsx`:

```typescript
const classifyWord = async (word: string) => {
  const response = await fetch('http://localhost:8000/classify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: word }),
  });
  
  if (!response.ok) {
    throw new Error('Classification failed');
  }
  
  const data = await response.json();
  
  // Transform backend response to frontend format
  return {
    predictions: data.prediction_results.map((result: any) => ({
      modelName: result.model,
      level: parseInt(result.prediction.split(' ')[1]) as 1 | 2 | 3 | 4 | 5 | 6,
      confidence: result.confidence,
    })),
    hardVote: data.ensemble_decision 
      ? parseInt(data.ensemble_decision.split(' ')[1]) as 1 | 2 | 3 | 4 | 5 | 6
      : 1,
  };
};
```

**Note**: Update the API URL to match your deployment (e.g., `https://your-backend-url.com/classify` for production).

## Other Endpoints

### `GET /`
Root endpoint with API information and list of loaded models.

### `GET /health`
Health check endpoint.

### `GET /docs`
Interactive API documentation (Swagger UI) - available when running the server.

## Preprocessing

The backend includes comprehensive Arabic text preprocessing:

1. **Language Detection**: Verifies input is Arabic using `langdetect` library
2. **Normalization**:
   - Removes diacritics (tashkeel)
   - Removes tatweel (elongation character)
   - Standardizes Alef variations (آ, أ, إ → ا)
   - Standardizes Ya variations (ى → ي)
   - Normalizes whitespace
3. **Tokenization**: Uses appropriate tokenizers for transformer models

## Model Requirements

### Classical ML Models

Your Hugging Face repository should contain:
- Model file: `.pkl` file with a scikit-learn compatible model (must have `predict()` or `predict_proba()` method)
- Optional: Vectorizer file: `.pkl` file with a vectorizer (must have `transform()` method)
- Optional: `config.json` with model metadata

Models can be packaged in a `.zip` file or as individual files.

### Transformer Models

Your Hugging Face repository should be a standard transformers model repository with:
- `config.json`: Model configuration
- Model weights: `.bin` or `.safetensors` files
- `tokenizer_config.json` and tokenizer files (auto-loaded)

The model should be a sequence classification model compatible with `AutoModelForSequenceClassification`.

## Error Handling

- **Non-Arabic text**: Returns 400 error with clear message
- **No models loaded**: Returns 503 error
- **Model prediction failures**: Logged but don't stop other models from running
- **Invalid input**: Returns 422 validation error

## Development

### Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py         # Model loading and inference
│   ├── preprocessing.py  # Arabic text preprocessing
│   └── config.py         # Configuration management
├── config.yaml           # Model configuration
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
└── README.md            # This file
```

## Troubleshooting

1. **Models not loading**: Check that your Hugging Face repository IDs are correct and accessible
2. **Authentication errors**: Set `HF_TOKEN` environment variable for private repositories
3. **Memory issues**: Large transformer models may require more RAM. Consider using smaller models or increasing container memory
4. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

Same as the main project.

