import json
import pandas as pd

# Load word frequency data
with open('./json/top_word_frequencies.json', 'r') as file:
    word_freq_data = json.load(file)

# Prepare a dictionary to store aggregated data
aggregated_data = {}

# Aggregate data across all models
for key, word_freq_list in word_freq_data.items():
    race, gender, _ = key.split('_')  # Split the key to get race and gender
    aggregated_key = f'{race}_{gender}'

    if aggregated_key not in aggregated_data:
        aggregated_data[aggregated_key] = []

    aggregated_data[aggregated_key].extend(word_freq_list)

# Prepare a list to store the statistics
word_freq_stats = []

# Calculate descriptive statistics for each combination of race and gender
for key, word_freq_list in aggregated_data.items():
    # Convert the list of [word, frequency] pairs to a DataFrame
    df = pd.DataFrame(word_freq_list, columns=['word', 'frequency'])

    # Group by word and sum the frequencies
    df = df.groupby('word').sum().reset_index()

    # Calculate statistics
    mean_freq = df['frequency'].mean()
    median_freq = df['frequency'].median()
    std_freq = df['frequency'].std()
    max_freq = df['frequency'].max()
    min_freq = df['frequency'].min()
    unique_words = df['word'].nunique()

    # Store the statistics in the list
    word_freq_stats.append({
        'key': key,
        'mean_frequency': mean_freq,
        'median_frequency': median_freq,
        'std_frequency': std_freq,
        'max_frequency': max_freq,
        'min_frequency': min_freq,
        'unique_words': unique_words
    })

# Convert the list of statistics to JSON format
word_freq_stats_json = json.dumps(word_freq_stats, indent=4)

# Save the statistics to a JSON file
with open('aggregated_word_freq_stats.json', 'w') as file:
    file.write(word_freq_stats_json)
