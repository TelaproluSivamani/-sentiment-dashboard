import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("reviews.csv")

df = df[['Review Text', 'Rating']].dropna()

# 3-class sentiment labeling
def label_sentiment(rating):
    if rating <= 2:
        return 0   # Negative
    elif rating == 3:
        return 1   # Neutral
    else:
        return 2   # Positive

df['sentiment'] = df['Rating'].apply(label_sentiment)
df.rename(columns={'Review Text': 'review'}, inplace=True)

print("Dataset loaded:", df.shape)

# -----------------------------
# Text Cleaning
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# -----------------------------
# Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------
# Model Training
# -----------------------------
from sklearn.svm import LinearSVC
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)



# -----------------------------
# Evaluation
# -----------------------------
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# -----------------------------
# Live Prediction
# -----------------------------
labels = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊"
}

while True:
    review = input("\nEnter a clothing review (or type 'exit'): ")
    if review.lower() == 'exit':
        break

    clean = clean_text(review)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]

    print("Sentiment:", labels[result])
