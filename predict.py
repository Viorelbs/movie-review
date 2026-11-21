import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent


def load_model():
    path = ROOT / "models" / "sentiment.joblib"
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python train.py")
    return joblib.load(path)


def analyze(model, text):
    text = text.strip()
    if not text:
        raise ValueError("Please enter a movie review")
    features = model.named_steps["tfidf"].transform([text])
    if features.nnz == 0:
        raise ValueError("No familiar words found. Please enter a review in English")
    scores = model.predict_proba([text])[0]
    best = int(scores.argmax())
    return {
        "label": str(model.classes_[best]),
        "score": float(scores[best]),
        "scores": {str(label): float(score) for label, score in zip(model.classes_, scores)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an English movie review")
    parser.add_argument("review", help="Movie review text in quotes")
    args = parser.parse_args()
    try:
        result = analyze(load_model(), args.review)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Prediction: {result['label']}")
    print(f"Model score: {result['score']:.1%}")
