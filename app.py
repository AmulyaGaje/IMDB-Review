import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------
# Page Config
# --------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

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
    color:#555;
    font-size:14px;
    letter-spacing:2px;
}

.metric-card{
    background:#d9f0ef;
    padding:20px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 3px 10px rgba(0,0,0,0.1);
}

.sentiment-tag{
    background:#ff9f1c;
    color:black;
    padding:5px 12px;
    border-radius:15px;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    background:#ff9f1c;
    color:black;
    border:none;
    border-radius:8px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# Load Models
# --------------------------------

model_rnn = tf.keras.models.load_model("simple_rnn_model.h5")
model_lstm = tf.keras.models.load_model("lstm_model.h5")
model_gru = tf.keras.models.load_model("gru_model.h5")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 200

# --------------------------------
# Header
# --------------------------------

st.markdown("""
<div class='header-box'>
<div class='header-title'>
Movie Review Sentiment Analysis System
</div>

<div class='header-sub'>
DEEP LEARNING BASED SENTIMENT CLASSIFICATION
</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------
# Model Selection
# --------------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN","LSTM","GRU"],
    horizontal=True
)

review = st.text_area(
    "Enter Your Review",
    height=150
)

# --------------------------------
# Prediction Function
# --------------------------------

def predict(model,text):

    seq = tokenizer.texts_to_sequences([text])

    pad = pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding='post'
    )

    pred = model.predict(
        pad,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if pred>=0.5
        else "Negative"
    )

    confidence = (
        pred if pred>=0.5
        else 1-pred
    )

    return sentiment,confidence,pred

# --------------------------------
# Analyze Button
# --------------------------------

if st.button("Analyze Review"):

    if review.strip()=="":

        st.warning("Enter a review")

    else:

        rnn_s,rnn_c,rnn_p = predict(
            model_rnn,
            review
        )

        lstm_s,lstm_c,lstm_p = predict(
            model_lstm,
            review
        )

        gru_s,gru_c,gru_p = predict(
            model_gru,
            review
        )

        # -------------------------
        # Cards
        # -------------------------

        st.subheader("Model Results")

        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>SimpleRNN</h4>
            <span class='sentiment-tag'>{rnn_s}</span>
            <h2>{rnn_c*100:.1f}%</h2>
            CONFIDENCE
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>LSTM</h4>
            <span class='sentiment-tag'>{lstm_s}</span>
            <h2>{lstm_c*100:.1f}%</h2>
            CONFIDENCE
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>GRU</h4>
            <span class='sentiment-tag'>{gru_s}</span>
            <h2>{gru_c*100:.1f}%</h2>
            CONFIDENCE
            </div>
            """, unsafe_allow_html=True)

        # -------------------------
        # Probability Chart
        # -------------------------

        st.subheader("Probability Distribution")

        chart_df = pd.DataFrame({
            "Model":["SimpleRNN","LSTM","GRU"],
            "Positive":[
                rnn_p*100,
                lstm_p*100,
                gru_p*100
            ],
            "Negative":[
                (1-rnn_p)*100,
                (1-lstm_p)*100,
                (1-gru_p)*100
            ]
        })

        fig = px.bar(
            chart_df,
            x="Model",
            y=["Positive","Negative"],
            barmode="group"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------
        # Gauge Charts
        # -------------------------

    for model_name, confidence, sentiment in [

    ("SimpleRNN", rnn_c*100, rnn_s),

    ("LSTM", lstm_c*100, lstm_s),

    ("GRU", gru_c*100, gru_s)

]:

        gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=confidence,

        number={'suffix': "%"},

        title={'text': model_name},

        gauge={

            'axis': {'range': [0,100]},

            'bar': {'thickness': 0.3},

            'steps': [

                {'range': [0,50], 'color': "#ffd6d6"},

                {'range': [50,100], 'color': "#d9f7e8"}

            ]
        }

    ))

    gauge.add_annotation(

        x=0.5,
        y=0.05,

        text=f"<b>{sentiment.upper()}</b>",

        showarrow=False,

        font=dict(
            size=18,
            color="#ff9f1c"
        )

    )

    gauge.update_layout(
        height=300,
        margin=dict(
            t=50,
            b=20,
            l=20,
            r=20
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.success(
            f"Selected Model ({selected_model}) Analysis Completed"
        )