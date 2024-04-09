from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import pandas as pd
from scipy import stats

# Standard Metric Analysis Statistical Tests
# Load the data into a pandas DataFrame
df = pd.read_json('./json/graph_data.json')

# List of unique races, genders, and metrics
races = df['race'].unique()
genders = df['gender'].unique()
metrics = ['sentiment', 'polarity', 'subjectivity', 'avg_response_length']

# Check normality for each race and metric combination
print("Normality Tests for Race:")
for race in races:
    for metric in metrics:
        stat, p_value = stats.shapiro(df[df['race'] == race][metric])
        print(f'Race: {race}, Metric: {metric}, P-value: {p_value}')

# Check normality for each gender and metric combination
print("\nNormality Tests for Gender:")
for gender in genders:
    for metric in metrics:
        stat, p_value = stats.shapiro(df[df['gender'] == gender][metric])
        print(f'Gender: {gender}, Metric: {metric}, P-value: {p_value}')

# Check equal variance for each metric across races
print("Equal Variance Tests for Race:")
for metric in metrics:
    race_groups = [df[df['race'] == race][metric] for race in races]
    stat, p_value = stats.levene(*race_groups)
    print(f'Metric: {metric}, P-value: {p_value}')

# Check equal variance for each metric across genders
print("\nEqual Variance Tests for Gender:")
for metric in metrics:
    male_scores = df[df['gender'] == 'Male'][metric]
    female_scores = df[df['gender'] == 'Female'][metric]
    stat, p_value = stats.levene(male_scores, female_scores)
    print(f'Metric: {metric}, P-value: {p_value}')

# Conduct one-way ANOVA for each metric
print("\nOne-Way ANOVA for Race:")
for metric in metrics:
    race_groups = [df[df['race'] == race][metric] for race in races]
    f_stat, p_value = stats.f_oneway(*race_groups)
    print(f'ANOVA for {metric}, P-value: {p_value}')

# Conduct two-sample t-tests for each metric
print("\nTwo-Sample T-Tests for Gender:")
for metric in metrics:
    male_scores = df[df['gender'] == 'Male'][metric]
    female_scores = df[df['gender'] == 'Female'][metric]
    t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
    print(f'T-test for {metric}, P-value: {p_value}')


# Lexical analysis statistical tests

lexical_df = pd.read_json('json/lexical_data.json')
# List of unique genders and lexical metrics
genders = lexical_df['gender'].unique()
lexical_metrics = ['male_verbs', 'male_adjectives',
                   'female_verbs', 'female_adjectives']

# Check normality for each gender and lexical metric combination
print("Normality Tests:")
for gender in genders:
    for metric in lexical_metrics:
        stat, p_value = stats.shapiro(
            lexical_df[lexical_df['gender'] == gender][metric])
        print(f'Gender: {gender}, Metric: {metric}, P-value: {p_value}')

# Check equal variance for each lexical metric across genders
print("\nEqual Variance Tests:")
for metric in lexical_metrics:
    male_scores = lexical_df[lexical_df['gender'] == 'Male'][metric]
    female_scores = lexical_df[lexical_df['gender'] == 'Female'][metric]
    stat, p_value = stats.levene(male_scores, female_scores)
    print(f'Metric: {metric}, P-value: {p_value}')

# Conduct two-sample t-tests for each lexical metric
print("\nTwo-Sample T-Tests for Gender:")
for metric in lexical_metrics:
    male_scores = lexical_df[lexical_df['gender'] == 'Male'][metric]
    female_scores = lexical_df[lexical_df['gender'] == 'Female'][metric]
    t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
    print(f'T-test for {metric}, P-value: {p_value}')

# Conduct paired t-tests for male vs female verbs/adjectives within each gender
print("\nPaired T-Tests Within Gender:")
for gender in genders:
    male_verbs_scores = lexical_df[lexical_df['gender']
                                   == gender]['male_verbs']
    female_verbs_scores = lexical_df[lexical_df['gender']
                                     == gender]['female_verbs']
    t_stat, p_value = stats.ttest_rel(male_verbs_scores, female_verbs_scores)
    print(f'Paired T-test for verbs, Gender: {gender}, P-value: {p_value}')

    male_adjectives_scores = lexical_df[lexical_df['gender']
                                        == gender]['male_adjectives']
    female_adjectives_scores = lexical_df[lexical_df['gender']
                                          == gender]['female_adjectives']
    t_stat, p_value = stats.ttest_rel(
        male_adjectives_scores, female_adjectives_scores)
    print(
        f'Paired T-test for adjectives, Gender: {gender}, P-value: {p_value}')

# Conduct Mann-Whitney U tests for each lexical metric since the data is not normally distributed
print("\nMann-Whitney U Test for Gender:")
for metric in lexical_metrics:
    male_scores = lexical_df[lexical_df['gender'] == 'Male'][metric]
    female_scores = lexical_df[lexical_df['gender'] == 'Female'][metric]
    u_stat, p_value = mannwhitneyu(male_scores, female_scores)
    print(f'Mann-Whitney U test for {metric}, P-value: {p_value}')


print("\nWilcoxon Signed-Rank Test Within Gender:")
for gender in genders:
    male_verbs_scores = lexical_df[lexical_df['gender']
                                   == gender]['male_verbs']
    female_verbs_scores = lexical_df[lexical_df['gender']
                                     == gender]['female_verbs']
    w_stat, p_value = wilcoxon(male_verbs_scores, female_verbs_scores)
    print(f'Wilcoxon test for verbs, Gender: {gender}, P-value: {p_value}')

    male_adjectives_scores = lexical_df[lexical_df['gender']
                                        == gender]['male_adjectives']
    female_adjectives_scores = lexical_df[lexical_df['gender']
                                          == gender]['female_adjectives']
    w_stat, p_value = wilcoxon(
        male_adjectives_scores, female_adjectives_scores)
    print(
        f'Wilcoxon test for adjectives, Gender: {gender}, P-value: {p_value}')
