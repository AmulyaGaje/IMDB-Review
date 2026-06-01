import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Models
# -----------------------------

model_rnn = tf.keras.models.load_model("simple_rnn_model.h5")
model_lstm = tf.keras.models.load_model("lstm_model.h5")
model_gru = tf.keras.models.load_model("gru_model.h5")

# -----------------------------
# Load Tokenizer
# -----------------------------

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 200

# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------

st.title("🎬 Movie Review Sentiment Analysis System")

st.markdown(
    "### Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# -----------------------------
# Model Selection
# -----------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# -----------------------------
# Input Area
# -----------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# -----------------------------
# Prediction Function
# -----------------------------

def predict_review(model, review_text):

    sequence = tokenizer.texts_to_sequences([review_text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if prediction >= 0.5
        else "Negative"
    )

    confidence = (
        prediction
        if prediction >= 0.5
        else 1 - prediction
    )

    return sentiment, confidence, prediction

# -----------------------------
# Analyze Review
# -----------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        if selected_model == "SimpleRNN":
            model = model_rnn

        elif selected_model == "LSTM":
            model = model_lstm

        else:
            model = model_gru

        sentiment, confidence, prob = predict_review(
            model,
            review
        )

        st.subheader("Prediction Result")

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        st.markdown("---")

        st.subheader("Visualization")

        positive_prob = prob * 100
        negative_prob = (1 - prob) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Positive Probability",
                f"{positive_prob:.2f}%"
            )

        with col2:
            st.metric(
                "Negative Probability",
                f"{negative_prob:.2f}%"
            )

        chart_df = pd.DataFrame(
            {
                "Probability": [
                    positive_prob,
                    negative_prob
                ]
            },
            index=[
                "Positive",
                "Negative"
            ]
        )

        st.bar_chart(chart_df)

# -----------------------------
# Compare All Models
# -----------------------------

st.markdown("---")

st.header("Compare Predictions From All Models")

if st.button("Compare Models"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        rnn_sent, rnn_conf, _ = predict_review(
            model_rnn,
            review
        )

        lstm_sent, lstm_conf, _ = predict_review(
            model_lstm,
            review
        )

        gru_sent, gru_conf, _ = predict_review(
            model_gru,
            review
        )

        comparison = pd.DataFrame({
            "Model": [
                "SimpleRNN",
                "LSTM",
                "GRU"
            ],

            "Sentiment": [
                rnn_sent,
                lstm_sent,
                gru_sent
            ],

            "Confidence (%)": [
                round(rnn_conf*100,2),
                round(lstm_conf*100,2),
                round(gru_conf*100,2)
            ]
        })

        st.dataframe(
            comparison,
            use_container_width=True
        )
