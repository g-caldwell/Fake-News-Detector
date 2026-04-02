# Fake News Detection System [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#) [![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Overview

The Fake News Detection System is a desktop application that classifies news articles as **real or fake** using machine learning and interactive data visualization.

The system integrates:

* A **data preprocessing pipeline**
* A **trained machine learning model**
* A **PyQt6 graphical user interface (GUI)**

It allows users to input text or datasets and receive predictions along with supporting analytics such as confidence scores, keyword insights, summaries, and readability metrics.

---

## Project Structure

```
Fake-News-Detector-main/
│
├── GUI/
│   ├── GUI.py
│   └── Assets/
│       ├── FakeNewsIcon.png
│       └── preview.gif
│
├── Model/
│   ├── clean.py
│   ├── model.py
│   ├── Cleaned Datasets/
│   ├── Raw Datasets/
│   └── Saved Models/
│       ├── fake_news_model.joblib
│       └── tfidf_vectorizer.joblib
│
├── README.md
└── LICENSE
```

---

## Objectives

* Classify news articles as **real or fake**
* Provide **explainable results** through visualizations and metrics
* Demonstrate **software engineering concepts** such as system modeling and architecture design

---

## System Components

### 1. Data Cleaning (`clean.py`)

* Processes raw news datasets
* Removes:

  * punctuation and special characters
  * URLs and social media references
  * irrelevant words (e.g., "reuters", "said")
* Outputs a cleaned dataset used for training

---

### 2. Machine Learning Model (`model.py`)

* Preprocesses cleaned text (tokenization, stopword removal)
* Converts text into numerical format using **TF-IDF vectorization**
* Trains a **Decision Tree Classifier**
* Saves:

  * trained model (`fake_news_model.joblib`)
  * vectorizer (`tfidf_vectorizer.joblib`)

---

### 3. Graphical User Interface (`GUI.py`)

Built using **PyQt6**, the GUI provides an interactive environment for users.

#### Features:

* Text input for manual article analysis
* File import (.txt)
* CSV upload for batch processing
* Real-time prediction display
* Visualization tools:

  * Pie chart (real vs fake confidence)
  * Bar chart (top keywords)
* Additional analytics:

  * AI explanation
  * Summary generation
  * Sentiment analysis
  * Readability scoring
* Dark/Light mode toggle

---

## System Workflow

### Model Training Pipeline

1. Load raw dataset
2. Clean data using `clean.py`
3. Preprocess text
4. Convert to TF-IDF vectors
5. Train Decision Tree model
6. Save trained model

---

### User Interaction Flow

1. User inputs text or uploads file/CSV
2. GUI sends input to detection system
3. Model predicts real or fake classification
4. System updates:

   * Prediction confidence chart
   * Keyword frequency chart
   * Summary and explanation
   * Sentiment and readability metrics

---

## Machine Learning Details

* **Vectorization:** TF-IDF
* **Model:** Decision Tree Classifier

  * Max Depth: 5
  * Min Samples Leaf: 15
* **Model Storage:**

  * `fake_news_model.joblib`
  * `tfidf_vectorizer.joblib`

---

## Features Summary

* Real vs Fake classification
* Text input, file upload, CSV batch processing
* Prediction confidence visualization
* Keyword extraction and analysis
* Text summarization
* Sentiment analysis
* Readability scoring
* Explainable AI output
* Dark/Light mode UI

---

## Software Engineering Concepts

This project applies key software engineering principles:

### System Modeling

* Context models (system boundaries and environment)
* Interaction models (user-system communication)
* Structural models (system architecture)
* Behavioral models (system workflow)

### Architecture Design

* Layered structure:

  * GUI (presentation layer)
  * Processing logic (application layer)
  * Data/model (data layer)

### UML Diagrams

* Use-case diagrams
* Sequence diagrams
* State diagrams
* Architectural diagrams

---

## How to Run

### 1. Install Dependencies

```bash
pip install pandas scikit-learn nltk matplotlib PyQt6 joblib
```

### 2. Clean Dataset

```bash
python Model/clean.py
```

### 3. Train Model

```bash
python Model/model.py
```

### 4. Run Application

```bash
python GUI/GUI.py
```

---

## Notes

* Pre-trained models are already included in `Saved Models/`
* Training step can be skipped if models are already available
* Large datasets are not included but can be added to `Raw Datasets/`

---

## Future Improvements

* Integrate advanced NLP models (e.g., transformers)
* Improve classification accuracy with larger datasets
* Deploy as a web or mobile application
* Enhance explainability with SHAP/LIME
* Add real-time news API integration

---

## Contributors


- [Gabriel Caldwell](https://www.github.com/g-caldwell)
- [Peyton Hollis](https://github.com/phollis11)
- [Tyler Wills](https://github.com/tylerwills27)
- [Carter Wilson](https://github.com/thanksyoungc)

---

## License

This project is licensed under the terms specified in the LICENSE file.

---
