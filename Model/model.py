import os
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

# File path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'Saved Models')
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'fake_news_model.joblib')
VECTORIZER_PATH = os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.joblib')

def download_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def preprocess_text(text_data):
    download_nltk()
    stop_words = stopwords.words('english')
    # Keeping it as a list to maintain exact original logic (token not in list)
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', str(sentence))
        preprocessed_text.append(' '.join(token.lower()
                                  for token in str(sentence).split()
                                  if token not in stop_words))
    return preprocessed_text

def train_and_save_model():
    dataset_path = os.path.join(BASE_DIR, 'Cleaned CSVs', 'dataset.csv')
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please run clean.py first.")
        return None, None

    data = pd.read_csv(dataset_path, index_col=0)
    data = data.sample(frac=1).reset_index(drop=True)
    
    preprocessed_review = preprocess_text(data['text'].values)
    data['text'] = preprocessed_review
    
    x_train, x_test, y_train, y_test = train_test_split(data['text'], 
                                                        data['class'], 
                                                        test_size=0.2)
    
    vectorization = TfidfVectorizer()
    x_train_vec = vectorization.fit_transform(x_train)
    x_test_vec = vectorization.transform(x_test)
    
    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=200, criterion='gini')
    model.fit(x_train_vec, y_train)
    
    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorization, VECTORIZER_PATH)
    
    print(f"Model trained and saved to {os.path.join(BASE_DIR, 'Saved Models')}")
    print(f"Training Accuracy: {accuracy_score(y_train, model.predict(x_train_vec))}")
    print(f"Testing Accuracy: {accuracy_score(y_test, model.predict(x_test_vec))}")
    return model, vectorization

def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return model, vectorizer
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
    return train_and_save_model()

def get_important_keywords(model, vectorizer, n=20):
    if not hasattr(model, 'feature_importances_'):
        return []
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    important_indices = importances.argsort()[::-1][:n]
    return [feature_names[i] for i in important_indices if importances[i] > 0]

if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()
    if model and vectorizer:
        keywords = get_important_keywords(model, vectorizer)
        print("Top keywords:", keywords)
