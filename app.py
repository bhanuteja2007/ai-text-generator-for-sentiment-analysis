import streamlit as st
from sentiment_model import detect_sentiment
from text_generator import generate_text

st.title("🧠 Sentiment-Aligned AI Text Generator")

prompt = st.text_area("Enter your prompt:")
manual_sentiment = st.selectbox("Choose sentiment (or leave as 'Auto')", ["Auto", "positive", "negative", "neutral"])
length = st.slider("Select output length (words)", 50, 300, 150)

if st.button("Generate"):
    sentiment = detect_sentiment(prompt) if manual_sentiment == "Auto" else manual_sentiment
    output = generate_text(prompt, sentiment, max_length=length)
    st.markdown(f"**Detected Sentiment:** {sentiment}")
    st.write(output)