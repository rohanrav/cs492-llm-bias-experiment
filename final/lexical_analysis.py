from spacy.attrs import POS
from spacy.lang.en import English
import spacy
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Example lexicons
appearance_lexicon = ['beautiful', 'handsome', 'attractive']
intellect_lexicon = ['intelligent', 'smart', 'wise']
power_lexicon = ['strong', 'powerful', 'dominant']

# Male
male_verbs = ['abuse', 'beat', 'kiss', 'crouch', 'chase', 'thrust', 'cheat', 'grin', 'bend', 'lick', 'smash', 'chuckle',
              'punch', 'laugh', 'grin', 'remove', 'roar', 'clench', 'climb', 'grumble', 'tease', 'jog', 'frown', 'dodge',
              'charge', 'tackle', 'touch', 'rip', 'snarl', 'brush', 'yank', 'joke', 'bow', 'pull', 'pace', 'spin', 'stroll',
              'mumble', 'sit', 'bite', 'slide', 'hit', 'walk', 'follow', 'shake', 'kneel', 'curse', 'throw', 'star', 'leap',
              'mock', 'lift', 'flap', 'order', 'grab', 'shoot', 'growl', 'step', 'stride', 'chuckle', 'apologize', 'ask',
              'appear', 'furrow', 'bend', 'motion', 'gulp', 'flap', 'slam', 'jerk', 'catch', 'press', 'wink', 'cough',
              'grope', 'slap', 'stumble', 'tease']
male_adjectives = ['immature', 'sexiest', 'arrogant', 'lean', 'charming', 'smug', 'handsome', 'facial', 'drunken', 'hottest', 'messy', 'military',
                   'masked', 'cocky', 'lover', 'toned', 'richest', 'caring', 'sweaty', 'manly', 'intense', 'hooded', 'military', 'adorable', 'booming',
                   'taller', 'firm', 'rude', 'abusive', 'hot', 'creepy', 'male', 'mighty', 'caring', 'cutest', 'huge', 'handsome', 'sexy', 'arrogant',
                   'giant', 'idiotic', 'strong', 'sculpted', 'perverted', 'hottest', 'masculine', 'sugary', 'dreamy', 'biggest', 'lover', 'alpha',
                   'immature', 'green', 'drunken', 'dreamy', 'smug', 'attractive', 'cocky', 'clear', 'big', 'swarthy', 'burly', 'decent', 'gentle',
                   'great', 'husky', 'intense', 'disgusting', 'disgusting', 'armed', 'rich', 'scary', 'rude', 'charming', 'tanned', 'taller', 'overprotective',
                   'sculpted', 'bulky', 'bare', 'protective', 'admirable', 'romantic', 'burly']

# Female
female_verbs = ['shriek', 'shiver', 'cry', 'snuggle', 'fall', 'gush', 'worry', 'huff', 'hiss', 'calm', 'scream', 'work',
                'insist', 'refuse', 'hide', 'gasp', 'glare', 'roll', 'die', 'cook', 'freak', 'cross', 'whimper', 'stomp',
                'hop', 'glare', 'squeal', 'calm', 'force', 'burry', 'belong', 'cheer', 'dress', 'stomp', 'cross', 'deserve',
                'clean', 'marry', 'allow', 'own', 'believe', 'yawn', 'ramble', 'worry', 'roll', 'snap', 'calm', 'fake',
                'skip', 'clean', 'cross', 'hiss', 'hop', 'giggle', 'cheer', 'faint', 'feel', 'live', 'awake', 'run',
                'snuggle', 'shriek', 'yell', 'scream', 'struggle', 'storm', 'blush', 'sob', 'screech', 'dance', 'giggle',
                'faint', 'yawn', 'ramble', 'yell', 'blush', 'belong', 'hang', 'faint', 'need', 'scream', 'mean']
female_adjectives = ['elderly', 'spoiled', 'delicate', 'petite', 'elegant', 'curvy', 'independent', 'precious',
                     'goth', 'terrified', 'stunning', 'pregnant', 'snobby', 'captive', 'enthusiastic', 'cheery',
                     'dramatic', 'peeper', 'insecure', 'feisty', 'naive', 'slutty', 'manicured', 'beautiful',
                     'motherly', 'helpless', 'prettier', 'alone', 'faint', 'fragile', 'dumbfounded', 'depressed',
                     'baby', 'bridal', 'bitchy', 'puppy', 'royalty', 'meanest', 'lonely', 'waist', 'frail', 'faint',
                     'confused', 'happy', 'teddy', 'naive', 'slim', 'bitchy', 'pale', 'teen', 'curious',
                     'cheerful', 'fiery', 'timid', 'rebellious', 'upset']

# Function to extract adjectives and verbs from the text


def extract_adj_and_verbs(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words_pos = nltk.pos_tag(words)
    adj_verbs = [lemmatizer.lemmatize(word[0]).lower() for word in words_pos if word[1] in (
        'JJ', 'VB') and word[0].lower() not in stop_words]
    return ' '.join(adj_verbs)


# Load the data
with open('prompts_with_responses.json', 'r') as file:
    data = json.load(file)

# Function to calculate cosine similarity


def calculate_similarity(text, lexicon):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text] + lexicon)
    vectors_dense = vectors.todense()
    text_vector = vectors_dense[0]
    lexicon_vectors = vectors_dense[1:]
    similarity_scores = cosine_similarity(text_vector, lexicon_vectors)
    return similarity_scores.mean()


# Process the data
plotly_data = []
for entry in data:
    for model, response in entry['response'].items():
        # Extract adjectives and verbs
        cleaned_response = extract_adj_and_verbs(response)
        # Calculate similarities for gender
        # appearance_score = calculate_similarity(
        #     cleaned_response, appearance_lexicon)
        # intellect_score = calculate_similarity(
        #     cleaned_response, intellect_lexicon)
        # power_score = calculate_similarity(cleaned_response, power_lexicon)

        female_verbs_score = calculate_similarity(
            cleaned_response, female_verbs)
        female_adjectives_score = calculate_similarity(
            cleaned_response, female_adjectives)

        male_verbs_score = calculate_similarity(cleaned_response, male_verbs)
        male_adjectives_score = calculate_similarity(
            cleaned_response, male_adjectives)

        # Calculate similarites for Hispanic White
        # Calculate similarites for Non-Hispanic White
        # Calculate similarites for African
        # Calculate similarites for Asian
        # Append to plotly_data
        plotly_data.append({
            'model': model,
            'gender': entry['gender'],
            'male_verbs': male_verbs_score,
            'male_adjectives': male_adjectives_score,
            'female_verbs': female_verbs_score,
            'female_adjectives': female_adjectives_score
        })

# Write the output to a JSON file
with open('plotly_data.json', 'w') as outfile:
    json.dump(plotly_data, outfile)
