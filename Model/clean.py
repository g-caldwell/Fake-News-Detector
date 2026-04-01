import pandas as pd
import re
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, 'News.csv')
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

if __name__ == "__main__":
    df = pd.read_csv(input_path)
    df['text'] = df['text'].apply(clean)
    df.to_csv(output_path, index=False)
    print(f"CSV saved to Cleaned CSVs/{output_path}")
