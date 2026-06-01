import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------------
# Custom CSS
# ----------------------------------

st.markdown("""
<style>

.header-box{
    background: linear-gradient(90deg,#2ec4b6,#a8dadc);
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-bottom:20px;
}

.header-title{
    font-size:32px;
    font-weight:bold;
    color:black;
}

.header-sub{
    font-size:14px;
    letter-spacing:2px;
    color:#555;
}

.metric-card{
    background:#d9f0ef;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
}

.sentiment-tag{
    background:#ff9f1c;
    color:black;
    padding:5px 15px;
    border-radius:20px;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    background:#ff9f1c;
    color:black;
    border:none;
    border-radius:8px;
    font-weight:bold;
    height:50px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load Models
# ----------------------------------

model_rnn = tf.keras.models.load_model("simple_rnn_model.h5")
model_lstm = tf.keras.models.load_model("lstm_model.h5")
model_gru = tf.keras.models.load_model("gru_model.h5")

# ----------------------------------
# Load Tokenizer
# ----------------------------------

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 200

# ----------------------------------
# Header
# ----------------------------------

st.markdown("""
<div class="header-box">
<div class="header-title">
🎬 Movie Review Sentiment Analysis System
</div>

<div class="header-sub">
DEEP LEARNING BASED SENTIMENT CLASSIFICATION
</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# Model Selection
# ----------------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"],
    horizontal=True
)

review = st.text_area(
    "Enter Your Review",
    height=180,
    placeholder="Enter your movie review here..."
)

# ----------------------------------
# Prediction Function
# ----------------------------------

def predict(model, text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
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

# ----------------------------------
# Analyze Button
# ----------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    # Select model

    if selected_model == "SimpleRNN":

        model = model_rnn

        model_desc = """
        SimpleRNN processes text sequentially using hidden states.
        It is fast and lightweight but may struggle to remember
        information from long reviews.
        """

    elif selected_model == "LSTM":

        model = model_lstm

        model_desc = """
        LSTM uses memory cells and gates to retain important
        information over long sequences. It performs well on
        long movie reviews.
        """

    else:

        model = model_gru

        model_desc = """
        GRU is an optimized recurrent network with fewer parameters
        than LSTM. It provides high accuracy while training faster.
        """

    sentiment, confidence, prob = predict(
        model,
        review
    )

    # ----------------------------------
    # Result Card
    # ----------------------------------

    st.subheader("Prediction Result")

    st.markdown(f"""
    <div class='metric-card'>
        <h3>{selected_model}</h3>
        <span class='sentiment-tag'>{sentiment}</span>
        <h1>{confidence*100:.2f}%</h1>
        CONFIDENCE
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------
    # Model Explanation
    # ----------------------------------

    st.markdown("---")

    st.subheader("Model Explanation")

    st.info(model_desc)

    # ----------------------------------
    # Sentiment Explanation
    # ----------------------------------

    st.subheader("Sentiment Explanation")

    if sentiment == "Positive":

        st.success(
            "The review contains positive words and expressions. "
            "The model predicts that the reviewer liked the movie."
        )

    else:

        st.error(
            "The review contains negative words and expressions. "
            "The model predicts that the reviewer disliked the movie."
        )

    # ----------------------------------
    # Probability Distribution
    # ----------------------------------

    positive_prob = prob * 100
    negative_prob = (1 - prob) * 100

    st.markdown("---")

    st.subheader("Probability Distribution")

    chart_df = pd.DataFrame({

        "Sentiment": ["Positive", "Negative"],

        "Probability": [
            positive_prob,
            negative_prob
        ]
    })

    fig = px.bar(
        chart_df,
        x="Sentiment",
        y="Probability",
        text="Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------------
    # Gauge Chart
    # ----------------------------------

    st.subheader("Confidence Gauge")

    gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=confidence * 100,

        number={"suffix":"%"},

        title={"text":selected_model},

        gauge={
            "axis":{"range":[0,100]},
            "bar":{"thickness":0.3}
        }

    ))

    sentiment_color = (
        "#2ec4b6"
        if sentiment == "Positive"
        else "#e63946"
    )

    gauge.add_annotation(
        x=0.5,
        y=0.05,
        text=f"<b>{sentiment.upper()}</b>",
        showarrow=False,
        font=dict(
            size=22,
            color=sentiment_color
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ----------------------------------
    # Compare All Models
    # ----------------------------------

    st.markdown("---")

    st.subheader("Compare All Models")

    rnn_s, rnn_c, _ = predict(model_rnn, review)
    lstm_s, lstm_c, _ = predict(model_lstm, review)
    gru_s, gru_c, _ = predict(model_gru, review)

    comparison = pd.DataFrame({

        "Model":[
            "SimpleRNN",
            "LSTM",
            "GRU"
        ],

        "Sentiment":[
            rnn_s,
            lstm_s,
            gru_s
        ],

        "Confidence (%)":[
            round(rnn_c*100,2),
            round(lstm_c*100,2),
            round(gru_c*100,2)
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )
