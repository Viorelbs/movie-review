# Movie Review Sentiment Analysis

A small Artificial Intelligence course project by Viorel Binciu

Enter an English movie review and predict whether it is positive or negative. The project uses TF-IDF features and logistic regression. The interface, code and report are in English.

## Quick start

Use Python 3.12 for the tested setup. Python 3.11 or newer is required by the pinned dependencies.

Open a terminal inside this folder:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
~~~

On Windows, activate the environment with `.venv\Scripts\activate` instead. Open the local URL printed by Streamlit, usually http://localhost:8501.

The trained model and dataset are included. Internet access is needed to install dependencies, but the demo then runs locally without API keys or paid services.

## Train the model again

~~~bash
python train.py
~~~

This replaces the model and evaluation files. The archive in `data/` is used when available; otherwise it is downloaded from Cornell. The report describes the bundled run and does not update automatically after retraining.

## Command-line prediction

~~~bash
python predict.py "A wonderful movie with excellent acting and a beautiful story."
~~~

## Method

1. Load Cornell sentiment polarity dataset v2.0: 2,000 English movie reviews
2. Create a stratified 80/20 train/test split with random seed 42
3. Fit a TF-IDF vectorizer on the 1,600 training reviews only
4. Train logistic regression using words and pairs of words
5. Evaluate once on the remaining 400 reviews
6. Save the pipeline for the interactive demo

TF-IDF uses up to 20,000 features, `min_df=2`, `max_df=0.95`, `ngram_range=(1, 2)` and sublinear term frequency. Common stop words are retained, including negation words. Logistic regression uses `C=1.0`, `solver="liblinear"` and `max_iter=1000`. No parameters were selected using the test results.

## Measured results

| Metric | Result |
| --- | --- |
| Training reviews | 1,600: 800 negative, 800 positive |
| Test reviews | 400: 200 negative, 200 positive |
| Test accuracy | 86.0%: 344 correct predictions |
| Majority-class baseline accuracy | 50.0% |
| Macro F1 | 0.860 |

| Actual / Predicted | Negative | Positive |
| --- | --- | --- |
| Negative | 171 | 29 |
| Positive | 27 | 173 |

These numbers describe one fixed holdout split, not every possible review. The app's model scores are uncalibrated estimates and are different from measured test accuracy.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `train.py` | Dataset loading, training and evaluation |
| `predict.py` | Shared prediction function and command-line demo |
| `requirements.txt` | Tested dependency versions |
| `models/sentiment.joblib` | Trained TF-IDF and classifier pipeline |
| `data/review_polarity.tar.gz` | Unchanged Cornell dataset archive |
| `data/README.md` | Dataset attribution |
| `results/metrics.json` | Metrics, feature terms and environment versions |
| `results/split.json` | Exact train/test review filenames |
| `results/errors.json` | Five misclassified test excerpts |
| `results/confusion_matrix.png` | Evaluation chart |
| `report.pdf` | Four-page project report |
| `report.md` | Editable report text |
| `presentation-notes.md` | Short script for presenting the project |

## Limits

- Only positive and negative classes are available; there is no neutral class
- The model was trained on older, mostly long English movie reviews
- Short text, sarcasm, mixed opinions and other languages can cause mistakes
- It learns word patterns rather than a full understanding of the plot or meaning
- One random split can differ from other splits; this is not the dataset paper's cross-validation protocol
- The data README notes that some rating information may remain in review text

## Troubleshooting

- If the model is missing, run `python train.py`
- If scikit-learn warns about a version mismatch, install the pinned requirements or retrain in your environment
- A review containing no known vocabulary will ask for another review
- Keep the project folder structure when copying or unzipping

Only load model files from trusted sources. A joblib model is a Python pickle-based file.

## References

- Course: Curs 1 - Notiuni fundamentale, Artificial Intelligence, 2022 edition
- Dataset v2.0: https://www.cs.cornell.edu/people/pabo/movie-review-data/
- Pang, B. and Lee, L. (2004). A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts. ACL
- TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Streamlit forms: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form
