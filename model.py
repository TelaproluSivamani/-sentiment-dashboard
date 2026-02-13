import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nltk.download('stopwords')

df = pd.read_csv("reviews.csv")
df = df[['Review Text', 'Rating']].dropna()

def label_sentiment(r):
    if r <= 2: return 0
    elif r == 3: return 1
    else: return 2

df['sentiment'] = df['Rating'].apply(label_sentiment)
df.rename(columns={'Review Text': 'review'}, inplace=True)

stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop_words])

df['clean'] = df['review'].apply(clean)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean'])
y = df['sentiment']

model = LinearSVC(class_weight='balanced')
model.fit(X, y)

labels = {0:"Negative", 1:"Neutral", 2:"Positive"}

def predict_sentiment(text):
    text = clean(text)
    vec = vectorizer.transform([text])
    return labels[model.predict(vec)[0]]
