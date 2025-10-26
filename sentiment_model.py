from textblob import TextBlob

def detect_sentiment(text):
    """
    Analyzes the text polarity using TextBlob and classifies it into 
    positive, negative, or neutral based on predefined thresholds.
    """
    if not text:
        return "neutral"
        
    polarity = TextBlob(text).sentiment.polarity
    
    # Classify sentiment based on polarity score
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"
