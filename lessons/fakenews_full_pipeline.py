

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#  Data Load 
df = pd.read_csv("data/train.csv")   # Ensure path is correct
print(df.head())

#  Clean Text
def clean_text(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

#  Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

#  Vectorization 
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#  Train Random Forest 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

#  Evaluation 
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#  Feature Importance 
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]
top_words = [vectorizer.get_feature_names_out()[i] for i in indices]
top_importances = importances[indices]

plt.figure(figsize=(10,6))
plt.barh(top_words, top_importances, color="purple")
plt.title("Top 20 Important Words for Fake News Prediction")
plt.xlabel("Importance")
plt.show()

#  Predict New News
new_news = [
    "Breaking: Scientists discover a new planet",
    "Celebrity caught in huge scandal!"
]

new_news_cleaned = [clean_text(text) for text in new_news]
new_news_vec = vectorizer.transform(new_news_cleaned)
predictions = model.predict(new_news_vec)

for text, pred in zip(new_news, predictions):
    label = "Fake" if pred==1 else "Real"
    print(f"News: {text}\nPrediction: {label}\n")

    
