import hashlib
import json
from pathlib import Path
import platform
import tarfile
import urllib.request

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
DATA_SHA256 = "fc0dccc2671af5db3c5d8f81f77a1ebfec953ecdd422334062df61ede36b2179"


def load_reviews():
    archive = ROOT / "data" / "review_polarity.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        print("Downloading the Cornell movie review dataset...")
        urllib.request.urlretrieve(DATA_URL, archive)
    actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_hash != DATA_SHA256:
        raise ValueError("Dataset checksum does not match the expected archive")
    texts, labels, names = [], [], []
    with tarfile.open(archive, "r:gz") as source:
        for member in sorted(source.getmembers(), key=lambda item: item.name):
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 3 or parts[0] != "txt_sentoken":
                continue
            if parts[1] not in {"pos", "neg"} or not parts[2].endswith(".txt"):
                continue
            with source.extractfile(member) as review:
                texts.append(review.read().decode("utf-8"))
            labels.append("positive" if parts[1] == "pos" else "negative")
            names.append(member.name)
    if len(texts) != 2000 or labels.count("positive") != 1000:
        raise ValueError("Expected 1,000 positive and 1,000 negative reviews")
    return texts, labels, names


def main():
    texts, labels, names = load_reviews()
    train_ids, test_ids = train_test_split(
        np.arange(len(texts)), test_size=0.2, random_state=42, stratify=labels
    )
    x_train = [texts[i] for i in train_ids]
    x_test = [texts[i] for i in test_ids]
    y_train = [labels[i] for i in train_ids]
    y_test = [labels[i] for i in test_ids]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_df=0.95, max_features=20000, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)),
    ])
    # Fit text features on training reviews only
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    class_order = ["negative", "positive"]
    matrix = confusion_matrix(y_test, predictions, labels=class_order)
    report = classification_report(y_test, predictions, labels=class_order, output_dict=True, zero_division=0)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    features = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    strongest = {
        "negative": features[np.argsort(weights)[:12]].tolist(),
        "positive": features[np.argsort(weights)[-12:][::-1]].tolist(),
    }
    metrics = {
        "dataset": "Cornell sentiment polarity dataset v2.0",
        "dataset_url": DATA_URL,
        "dataset_sha256": DATA_SHA256,
        "random_state": 42,
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "train_class_counts": {label: y_train.count(label) for label in class_order},
        "test_class_counts": {label: y_test.count(label) for label in class_order},
        "features": len(features),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "baseline_accuracy": max(y_test.count(label) for label in class_order) / len(y_test),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_order": class_order,
        "strongest_features": strongest,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    for folder in ["models", "results"]:
        (ROOT / folder).mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "sentiment.joblib", compress=3)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    split = {"train": [names[i] for i in train_ids], "test": [names[i] for i in test_ids]}
    (ROOT / "results" / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    examples = [
        {"file": names[i], "actual": actual, "predicted": predicted, "excerpt": texts[i][:550]}
        for i, actual, predicted in zip(test_ids, y_test, predictions) if actual != predicted
    ][:5]
    (ROOT / "results" / "errors.json").write_text(json.dumps(examples, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.1, 4.5), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=class_order).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title("Performance on 400 unseen reviews", pad=18)
    fig.savefig(ROOT / "results" / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    print(classification_report(y_test, predictions, digits=3))
    print(f"Accuracy: {metrics['accuracy']:.1%}; baseline: {metrics['baseline_accuracy']:.1%}")
    print("Saved the model and evaluation results")


if __name__ == "__main__":
    main()
