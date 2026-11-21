import json
from pathlib import Path

import streamlit as st

from predict import analyze, load_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


st.title("Movie Review Sentiment Analysis")
st.write("What did you think of the movie? Write a review in English and let the model predict its sentiment.")

try:
    model = get_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

examples = {
    "Write my own review": "",
    "Positive example": "A wonderful movie with excellent acting and a beautiful story. I loved every minute of it.",
    "Negative example": "A terrible movie with boring characters and bad acting. The story was predictable and a waste of time.",
    "Mixed example": "The acting was excellent, but the story was boring and the ending was disappointing.",
}


def use_example():
    st.session_state.review = examples[st.session_state.example]
    st.session_state.pop("result", None)


st.selectbox("Try an example", list(examples), key="example", on_change=use_example)
with st.form("review_form"):
    review = st.text_area("Your movie review", key="review", height=170, placeholder="The movie was...")
    submitted = st.form_submit_button("Analyze review", type="primary")

if submitted:
    st.session_state.pop("result", None)
    try:
        result = analyze(model, review)
        st.session_state.result = result
    except ValueError as error:
        st.warning(str(error))

if "result" in st.session_state:
    result = st.session_state.result
    if result["label"] == "positive":
        st.success("Prediction: Positive")
    else:
        st.error("Prediction: Negative")
    left, right = st.columns(2)
    left.metric("Positive score", f"{result['scores']['positive']:.1%}")
    right.metric("Negative score", f"{result['scores']['negative']:.1%}")
    st.caption("Scores are model estimates, not measured accuracy. Mixed and neutral reviews still receive one of the two labels.")

with st.expander("How it works"):
    st.write("1. TF-IDF turns words and word pairs into numeric features.")
    st.write("2. Logistic regression uses patterns learned from labeled movie reviews.")
    st.write("3. The model returns a positive or negative prediction.")
    st.write("It can make mistakes with sarcasm, short reviews and unfamiliar language.")

with st.expander("Evaluation"):
    path = ROOT / "results" / "metrics.json"
    if path.exists():
        metrics = json.loads(path.read_text())
        left, right = st.columns(2)
        left.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        right.metric("Test reviews", metrics["test_size"])
        st.caption(f"Training reviews: {metrics['train_size']} · Majority baseline: {metrics['baseline_accuracy']:.0%}")
        chart = ROOT / "results" / "confusion_matrix.png"
        if chart.exists():
            st.image(str(chart))
    else:
        st.info("Run python train.py to generate evaluation results")

st.caption("AI course project · TF-IDF + Logistic Regression · Cornell movie review dataset v2.0")
