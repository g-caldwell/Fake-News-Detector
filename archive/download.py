import os
import pandas as pd
import kagglehub
import shutil

# Paths
# This script is in archive/, so parent is root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'Model', 'Raw Datasets')

# Dataset 1: Bigg Data Sets (300k)
BIGG_DATASET = {
    "handle": "biggdatasets/fake-and-real-news-dataset",
    "filename": "Fake and Real News Dataset.xlsx",
    "output": "News.csv"
}

# Dataset 2: ISOT / Clement Bisaillon (44k) - The "Standard" version
ISOT_DATASET = {
    "handle": "clmentbisaillon/fake-and-real-news-dataset",
    "output": "News_44k.csv"
}

def setup_bigg_dataset():
    excel_path = os.path.join(ARCHIVE_DIR, BIGG_DATASET["filename"])
    output_path = os.path.join(RAW_DATA_DIR, BIGG_DATASET["output"])
    
    if not os.path.exists(excel_path):
        print(f"Downloading Bigg dataset {BIGG_DATASET['handle']}...")
        try:
            path = kagglehub.dataset_download(BIGG_DATASET["handle"])
            for f in os.listdir(path):
                if f.endswith('.xlsx'):
                    shutil.copy(os.path.join(path, f), excel_path)
                    print(f"File saved to {excel_path}")
                    break
        except Exception as e:
            print(f"Error downloading Bigg: {e}")
            return

    if os.path.exists(excel_path):
        print(f"Converting Bigg Excel to CSV: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # Mapping
        label_col = next((c for c in df.columns if c.lower() in ['label', 'news_type']), None)
        if label_col:
            df['class'] = df[label_col].apply(lambda x: 1 if str(x).lower() == 'real' else 0)
        
        text_col = next((c for c in df.columns if c.lower() in ['text', 'full_text']), None)
        if text_col:
            df = df.rename(columns={text_col: 'text'})
        
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Bigg dataset prepared at {output_path} ({len(df)} rows)")

def setup_isot_dataset():
    output_path = os.path.join(RAW_DATA_DIR, ISOT_DATASET["output"])
    
    print(f"Downloading ISOT dataset {ISOT_DATASET['handle']}...")
    try:
        path = kagglehub.dataset_download(ISOT_DATASET["handle"])
        # This dataset has True.csv and Fake.csv
        true_file = os.path.join(path, "True.csv")
        fake_file = os.path.join(path, "Fake.csv")
        
        if os.path.exists(true_file) and os.path.exists(fake_file):
            print("Reading ISOT files...")
            df_true = pd.read_csv(true_file)
            df_fake = pd.read_csv(fake_file)
            
            # Add labels
            df_true['class'] = 1
            df_fake['class'] = 0
            
            # Combine
            df = pd.concat([df_true, df_fake], ignore_index=True)
            
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"ISOT dataset prepared at {output_path} ({len(df)} rows)")
        else:
            print("Required CSV files not found in ISOT download.")
    except Exception as e:
        print(f"Error downloading/processing ISOT: {e}")

if __name__ == "__main__":
    print("--- Setting up 300k Bigg Dataset ---")
    setup_bigg_dataset()
    print("\n--- Setting up 44k ISOT Dataset ---")
    setup_isot_dataset()
