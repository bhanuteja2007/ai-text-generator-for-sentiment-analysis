Sentiment-Aligned AI Text Generator
***Overview
This project is a web-based AI application that generates paragraphs or essays based on the sentiment of a user’s input prompt. It uses sentiment analysis to classify the input as positive, negative, or neutral, and then generates text aligned with that sentiment using a pre-trained language model.

***Features:-
- Automatic sentiment detection from user input
- AI-generated text matching the detected sentiment
- Optional manual sentiment selection
- Adjustable output length (word count)
- Interactive frontend built with Streamlit
***Tech Stack:-
- Python: The core programming language used for backend logic and integration.
- TextBlob: A simple NLP library used for sentiment analysis of the input prompt.
- Hugging Face Transformers (GPT-2): A pre-trained language model used to generate text aligned with the detected sentiment.
- Torch (PyTorch): Required for running transformer models efficiently.
- Streamlit: A Python-based framework used to build the interactive web frontend.
- Virtual Environment (venv): Used to isolate project dependencies.
- Streamlit Cloud or Heroku (optional): Platforms for deploying the app online.
- Git & GitHub: For version control and sharing your code repository.
- Markdown (README.md): For documenting your project setup, methodology, and usage instructions.
***Project Structure:-
ai-text-generator/
->app.py                
-> sentiment_model.py 
-> text_generator.py       
->requirements.txt   
->README.md 
***Installation & Setup
- Clone the repository
	git clone https://github.com/your-username/ai text generator.git
	cd ai text generator
- Create and activate a virtual environment
	python -m venv venv
	venv\Scripts\activate  # Windows
- Install dependencies
	pip install -r requirements.txt
- Run the app
	streamlit run app.py
***How It Works:-
1. Sentiment Detection
- Uses TextBlob to analyze the polarity of the input prompt.
- Classifies sentiment as:
- positive if polarity > 0.1
- negative if polarity < -0.1
- neutral otherwise
2. Text Generation
- Uses GPT-2 via Hugging Face’s transformers pipeline.
- Prepends a sentiment-specific prefix to guide the generation:
- Positive → “Here’s a cheerful take:”
- Negative → “Here’s a critical view:”
- Neutral → “Here’s a balanced perspective:”
***Deployment (Optional)
Streamlit Cloud
- Push your code to GitHub.
- Go to https://streamlit.io/cloud.
- Connect your GitHub repo and deploy.
Heroku (Alternative)
- Add a Procfile:
web: streamlit run app.py
- Push to Heroku using Git.
***Optional Enhancements
- Manual sentiment override via dropdown
- Word count slider for output length
- Essay vs paragraph toggle
- Multilingual support using other Hugging Face models
- Custom fine-tuned GPT-2 for stronger sentiment control
***Challenges Faced
- Ensuring sentiment alignment in generated text
- Handling ambiguous or mixed-sentiment prompts
- Balancing coherence with sentiment-specific prefixes
- Managing model response length and relevance
***License
This project is open-source and available under the MIT License.


