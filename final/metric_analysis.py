import json
import re
from textblob import TextBlob

# Load data
with open('prompts_with_responses.json', 'r') as file:
    data = json.load(file)

# Preprocess data
preprocessed_data = []
for entry in data:
    race = entry['race']
    gender = entry['gender']
    for model, response in entry['response'].items():
        # Clean response
        clean_response = re.sub(r'[^a-zA-Z\s]', '', response).lower()
        # Calculate sentiment, polarity, and subjectivity
        blob = TextBlob(clean_response)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        # Calculate average response length
        avg_response_length = len(clean_response.split())
        preprocessed_data.append({
            'race': race,
            'gender': gender,
            'model': model,
            'sentiment': sentiment,
            'polarity': sentiment,  # Polarity is the same as sentiment in TextBlob
            'subjectivity': subjectivity,
            'avg_response_length': avg_response_length
        })

# Save preprocessed data to JSON file
with open('preprocessed_data.json', 'w') as outfile:
    json.dump(preprocessed_data, outfile)
