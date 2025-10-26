import streamlit as st
from transformers import pipeline
from sentiment_model import detect_sentiment
from text_generator import generate_text

# --- CRITICAL FIX: Caching the Model ---
@st.cache_resource
def get_generator():
    """Initializes and caches the Hugging Face text-generation pipeline."""
    with st.spinner("Initializing AI model (This runs only once)..."):
        return pipeline("text-generation", model="gpt2")

# Initialize the generator
try:
    generator = get_generator()
except Exception as e:
    st.error(f"Failed to load AI model. Please check dependencies. Error: {e}")
    generator = None

st.title("🧠 Sentiment-Aligned AI Text Generator")

st.markdown("""
This application uses TextBlob for sentiment analysis and GPT-2 to generate text that matches the required sentiment.
""")

prompt = st.text_area("Enter your prompt (e.g., 'The future of quantum computing will be...')", height=100)
manual_sentiment = st.selectbox("Choose sentiment (or leave as 'Auto')", ["Auto", "positive", "negative", "neutral"])
# Updated UI label for accuracy: 'max_length' is measured in tokens, not words
length = st.slider("Select output length (tokens)", 50, 500, 150)

if st.button("Generate"):
    if not prompt:
        st.warning("Please enter a prompt to begin generation.")
    elif generator is None:
        st.error("The AI model is not available.")
    else:
        # 1. Sentiment Detection
        sentiment = detect_sentiment(prompt) if manual_sentiment == "Auto" else manual_sentiment

        # 2. Text Generation
        with st.spinner(f'Generating text with a {sentiment.upper()} alignment...'):
            try:
                # Pass the cached generator instance to the generation function
                output = generate_text(prompt, sentiment, max_length=length, generator=generator)

                # --- FIX FOR PARAGRAPH MODE ---
                # 1. Replace all newlines with a single space.
                # 2. Remove extra spaces to ensure clean paragraph formatting.
                clean_output = output.replace('\n', ' ').strip()
                clean_output = ' '.join(clean_output.split())
                
                # 3. Output Results
                st.markdown("---")
                # Using HTML for colored text for better visual feedback
                st.markdown(f"**💡 Detected Sentiment:** <span style='color: #1E88E5; font-weight: bold;'>{sentiment.upper()}</span>", unsafe_allow_html=True)
                st.subheader("Generated Content")
                
                # Use st.markdown for standard paragraph display
                st.markdown(clean_output) 

            except Exception as e:
                st.error(f"An error occurred during text generation. Error: {e}")

st.markdown("---")
st.caption("Model: GPT-2 via Hugging Face Transformers | Sentiment: TextBlob")
