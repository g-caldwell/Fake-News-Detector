import pandas as pd
import re
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, 'Raw Datasets')
input_path = os.path.join(input_dir, 'News.csv')
output_dir = os.path.join(BASE_DIR, 'Cleaned CSVs')
output_path = os.path.join(output_dir, 'dataset.csv')


def clean(text):
    text = str(text).lower()
    
    # Remove reuters and locations
    text = re.sub(r'^.*?\(reuters\)\s*[-—:]', '', text)
    text = re.sub(r'^\s*\w+\s+[-—:]', '', text)
    
    # Social media @'s and websites
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # List of words to remove
    filteredWords = [
        'said', 'told', 'reported', 'spokesman', 'according', 
        'image', 'images', 'via', 'featured', 'video', 'watch', 'read',
        'reuters', 'facebook', 'twitter', 'getty', 'photo', 'by', 
        'screenshot', 'screen capture', 'screencapture'
    ]
    pattern = re.compile(r'\b(' + '|'.join(filteredWords) + r')\b')
    text = pattern.sub('', text)

    # Remove punctuation and double spaces at the end
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_csv(path_in, path_out):
    """Cleans a CSV file using the clean function on the 'text' column."""
    if not os.path.exists(path_in):
        print(f"Error: {path_in} does not exist.")
        return False
        
    df = pd.read_csv(path_in)
    if 'text' not in df.columns:
        print(f"Error: 'text' column not found in {path_in}.")
        return False
        
    df['text'] = df['text'].apply(clean)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    
    df.to_csv(path_out, index=False)
    print(f"CSV saved to: {path_out}")
    return True


if __name__ == "__main__":
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        exit()
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all CSVs in Raw Datasets
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            in_path = os.path.join(input_dir, filename)
            out_file = "dataset.csv" if filename == "News.csv" else f"dataset_{filename.replace('News_', '')}"
            out_path = os.path.join(output_dir, out_file)
            
            print(f"Cleaning {filename}...")
            clean_csv(in_path, out_path)
