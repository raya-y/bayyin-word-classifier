# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import gdown
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Download the model from Google Drive if not exists
MODEL_PATH = "rf_best_halving_nsamples.pkl"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1iMsPs7NVvAA2qnK9lvmJetRfirnXTAns"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

# Optional: you might need the vectorizer if the model expects transformed features
# e.g., tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Input format
class WordInput(BaseModel):
    word: str

@app.post("/classify")
def classify_word(input: WordInput):
    word = input.word

    # Transform the word (adjust according to your model preprocessing)
    # Example: vector = tfidf_vectorizer.transform([word])
    # If your model expects raw text, skip vectorization
    prediction = rf_model.predict([word])[0]  # adjust if needed
    confidence = rf_model.predict_proba([word])[0].max() if hasattr(rf_model, "predict_proba") else 1.0

    # Level schedule
    level = int(prediction)  # make sure your model outputs a numeric level 1-6

    return {
        "predictions": [
            {
                "modelName": "Random Forest",
                "level": level,
                "confidence": confidence
            }
        ],
        "hardVote": level
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
