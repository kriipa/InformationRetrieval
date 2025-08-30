import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = './engine/classifier_model.joblib'

def load_data_from_csv_files(base_path):
    documents = []
    labels = []
    category_map = {
        'politics.csv': 'politics',
        'business.csv': 'business',
        'health.csv': 'health'
    }
    for filename in os.listdir(base_path):
        if filename.endswith('.csv') and filename in category_map:
            file_path = os.path.join(base_path, filename)
            category = category_map[filename]
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                if 'Headline' in df.columns:
                    for headline in df['Headline'].fillna('').tolist():
                        documents.append(headline)
                        labels.append(category)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return documents, labels

def train_classifier(data_path='../data/'):
    X, y = load_data_from_csv_files(data_path)
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', LogisticRegression(random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

def get_or_train_classifier(data_path='../data/'):
    if os.path.exists(MODEL_PATH):
        print("Loading classifier model from disk...")
        return joblib.load(MODEL_PATH)
    else:
        print("Training new classifier model...")
        model = train_classifier(data_path)
        joblib.dump(model, MODEL_PATH)
        print("Classifier model saved to disk.")
        return model
