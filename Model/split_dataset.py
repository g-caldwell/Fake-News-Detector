import os
import pandas as pd

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Cleaned CSVs', 'dataset.csv')
output_dir = os.path.join(BASE_DIR, 'Split_Data')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def split_data():
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find {dataset_path}. Please run clean.py first.")
        return

    print(f"Reading {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Filter for Fake (class 0) and Real (class 1)
    fake_df = df[df['class'] == 0]
    real_df = df[df['class'] == 1]

    fake_output = os.path.join(output_dir, 'fake_news.csv')
    real_output = os.path.join(output_dir, 'real_news.csv')

    print(f"Saving {len(fake_df)} rows to {fake_output}...")
    fake_df.to_csv(fake_output, index=False)

    print(f"Saving {len(real_df)} rows to {real_output}...")
    real_df.to_csv(real_output, index=False)

    print("Success! Dataset split correctly.")

if __name__ == "__main__":
    split_data()
