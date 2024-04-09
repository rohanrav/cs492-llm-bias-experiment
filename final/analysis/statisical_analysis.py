import json
import pandas as pd

# Load preprocessed data
with open('./json/graph_data.json', 'r') as file:
    preprocessed_data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(preprocessed_data)

# Compute descriptive statistics aggregated by race and gender
race_aggregated_stats = df.groupby(['race']).agg({
    'sentiment': ['mean', 'median', 'std', 'min', 'max'],
    'polarity': ['mean', 'median', 'std', 'min', 'max'],
    'subjectivity': ['mean', 'median', 'std', 'min', 'max'],
    'avg_response_length': ['mean', 'median', 'std', 'min', 'max']
}).reset_index()

# Load lexical analysis data
with open('./json/lexical_data.json', 'r') as file:
    lexical_data = json.load(file)

# Convert to DataFrame
lexical_df = pd.DataFrame(lexical_data)

# Compute descriptive statistics for lexical analysis metrics aggregated by gender
lexical_race_aggregated_stats = lexical_df.groupby(['gender']).agg({
    'male_verbs': ['mean', 'median', 'std', 'min', 'max'],
    'male_adjectives': ['mean', 'median', 'std', 'min', 'max'],
    'female_verbs': ['mean', 'median', 'std', 'min', 'max'],
    'female_adjectives': ['mean', 'median', 'std', 'min', 'max']
}).reset_index()

# Convert the statistics to JSON format
race_aggregated_stats_json = race_aggregated_stats.to_json(orient='records')
lexical_race_aggregated_stats_json = lexical_race_aggregated_stats.to_json(
    orient='records')

# Save the statistics to JSON files
with open('race_aggregated_stats.json', 'w') as file:
    file.write(race_aggregated_stats_json)

# with open('lexical_race_aggregated_stats.json', 'w') as file:
#     file.write(lexical_race_aggregated_stats_json)
