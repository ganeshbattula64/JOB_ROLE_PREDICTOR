import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean text
df["resume_text"] = df["resume_text"].astype(str).apply(preprocess_text)

# Features and labels
X = df["resume_text"]
y = df["job_role"]  # Make sure 'job_role' column exists in dataset.csv

# Define pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X, y)

# Save model
joblib.dump(model_pipeline, "job_role_model.pkl")
print("✅ Model trained and saved as job_role_model.pkl")
