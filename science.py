from responses import responses
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


def get_word_frequency(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower()
                      not in stopwords.words('english')]
    return Counter(filtered_words)


for response in responses:
    response['sentiment'] = get_sentiment(response['response'])
    response['word_frequency'] = get_word_frequency(response['response'])
    response['text_length'] = len(response['response'].split())

data_by_profile = {}
for response in responses:
    profile = response['profile']
    if profile not in data_by_profile:
        data_by_profile[profile] = {'sentiment': [], 'text_length': []}
    data_by_profile[profile]['sentiment'].append(response['sentiment'])
    data_by_profile[profile]['text_length'].append(response['text_length'])

plt.figure(figsize=(10, 6))
for profile, data in data_by_profile.items():
    plt.scatter(data['sentiment'], data['text_length'], label=profile)
plt.xlabel('Sentiment Polarity')
plt.ylabel('Text Length')
plt.title('Sentiment Polarity vs Text Length by Profile')
plt.legend()
plt.show()
