# /c:/python/c/text_classification.py

import re
import random
from pathlib import Path
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEM = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [LEM.lemmatize(t) for t in tokens if t not in STOP_WORDS and t.isalpha()]
    return " ".join(tokens)


def build_dataset() -> List[Tuple[str, str]]:
    # small sample, replace with real corpus/file
    return [
        ("I love this product, it is fantastic!", "positive"),
        ("This is terrible and I hate it", "negative"),
        ("Can you buy this now? great value", "positive"),
        ("Spam message: click this link for free money", "spam"),
        ("Hello friend, are we meeting later?", "ham"),
        ("You won a prize! claim now", "spam"),
        ("The movie was boring and too long", "negative"),
        ("Excellent service and friendly staff", "positive"),
        ("Don't miss this opportunity", "spam"),
        ("I am very happy with the results", "positive"),
    ]


def train_and_save_model():
    data = build_dataset()
    texts, labels = zip(*data)
    
    # FIX: Clean texts BEFORE train/test split
    cleaned_texts = [clean_text(text) for text in texts]
    
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # FIX: Remove preprocessor parameter
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    out = Path("text_classifier.pkl")
    import joblib

    joblib.dump(pipeline, out)
    print("Model saved:", out)


def predict(texts: List[str]):
    import joblib

    model = joblib.load("text_classifier.pkl")
    # FIX: Clean input texts before prediction
    cleaned_texts = [clean_text(text) for text in texts]
    preds = model.predict(cleaned_texts)
    for t, p in zip(texts, preds):
        print(f"[{p}] {t}")


if __name__ == "__main__":
    train_and_save_model()
    sample = [
        "This email offers you a great deal on new phones",
        "I had an amazing dinner last night",
        "Please call me back, it's urgent",
    ]
    predict(sample)