import pandas as pd
import random

# Load data from CSV
df = pd.read_csv('events.csv')

# Function to shuffle each column independently
def shuffle_columns(dataframe):
    shuffled_df = dataframe.copy()
    for column in dataframe.columns:
        shuffled_df[column] = random.sample(dataframe[column].tolist(), len(dataframe[column]))
    return shuffled_df

# Generate synthetic data
synthetic_df = shuffle_columns(df)

# Optional: Save synthetic data to a new CSV
synthetic_df.to_csv('synthetic_events.csv', index=False)
