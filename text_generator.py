from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, sentiment, max_length=150):
    prefix = {
        "positive": "Here's a cheerful take: ",
        "negative": "Here's a critical view: ",
        "neutral": "Here's a balanced perspective: "
    }
    input_text = prefix[sentiment] + prompt
    result = generator(input_text, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']