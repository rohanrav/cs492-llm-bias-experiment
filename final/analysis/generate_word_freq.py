import json
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Set of English stopwords
stop_words = set(stopwords.words('english'))


def clean_and_tokenize(text):
    # Convert to lowercase, remove punctuation, and tokenize
    text = text.lower()
    text = re.sub(rf'[{string.punctuation}]', '', text)
    tokens = word_tokenize(text)
    # Remove stopwords and single-character tokens (mostly punctuation)
    tokens = [
        word for word in tokens if word not in stop_words and len(word) > 1]
    return tokens


# Load data
with open('prompts_with_responses.json', 'r') as file:
    data = json.load(file)

# Initialize dictionaries for aggregate data
aggregate_all = Counter()
aggregate_by_race = {}
aggregate_by_gender = {}
aggregate_by_model = {}

# Dictionary to hold top word frequencies for each category
top_word_freq = {}

# Initialize counters for races and genders
races = set(entry['race'] for entry in data) | {"All_races"}
genders = set(entry['gender'] for entry in data) | {"All_genders"}
models = set(model for entry in data for model in entry['response']) | {
    "All_models"}

for race in races:
    aggregate_by_race[race] = Counter()
for gender in genders:
    aggregate_by_gender[gender] = Counter()
for model in models:
    aggregate_by_model[model] = Counter()

# Process the data
for entry in data:
    race = entry['race']
    gender = entry['gender']
    for model, response in entry['response'].items():
        # Clean and tokenize the response
        tokens = clean_and_tokenize(response)

        # Update specific and aggregate counters
        keys = [(race, gender, model),
                (race, gender, "All models"),
                (race, "All genders", "All models"),
                (race, "All genders", model),
                ("All races", gender, model),
                ("All races", gender, "All models"),
                ("All races", "All genders", model)]

        for key in keys:
            key_str = '_'.join(key)
            if key_str not in top_word_freq:
                top_word_freq[key_str] = Counter()
            top_word_freq[key_str].update(tokens)

        # Update overall aggregate
        aggregate_all.update(tokens)

# Extract top 15 words for each counter
for key, counter in top_word_freq.items():
    top_word_freq[key] = counter.most_common(15)

# Add overall aggregate to the final JSON
top_word_freq["All races_All genders_All models"] = aggregate_all.most_common(
    15)

# Save the word frequency data to a JSON file
with open('top_word_frequencies.json', 'w') as outfile:
    json.dump(top_word_freq, outfile, indent=4)
