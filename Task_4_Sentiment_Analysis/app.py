# =========================================
# SENTIMENT ANALYSIS USING Twitter_Data_Cleaned.csv
# =========================================

# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv("C:\OASIS\PROJECT4\Twitter_Data_Cleaned.csv")  # Use raw string for Windows

print("Dataset Shape:", df.shape)
print(df.head())

# ===============================
# 2a. Handle Missing Values
# ===============================
# Drop rows where 'clean_text' or 'category' is NaN
df = df.dropna(subset=['clean_text', 'category'])

# Ensure category column is integer (or string if desired)
df['category'] = df['category'].astype(int)

# ===============================
# 3. Text Cleaning (NLP Preprocessing)
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\S+", "", text)     # Remove mentions
    text = re.sub(r"#\S+", "", text)     # Remove hashtags
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove numbers
    text = text.strip()
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

# ===============================
# 4. Feature Engineering (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# ===============================
# 5. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. Train Model (Naive Bayes)
# ===============================
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb))

# ===============================
# 7. Train Model (SVM)
# ===============================
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm))

# ===============================
# 8. Confusion Matrix Visualization
# ===============================
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), 
            annot=True, 
            fmt="d", 
            cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 9. Sentiment Distribution Visualization
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x=df['category'])
plt.title("Sentiment Distribution")
plt.show()

# ===============================
# 10. Custom Prediction Function
# ===============================
def predict_sentiment(text):
    """
    Predict the sentiment of a new tweet.
    Returns the predicted category.
    """
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = svm_model.predict(vector)
    return prediction[0]

# Example usage
sample_text = "I love the new features in this app!"
print("\nCustom Text Prediction:", predict_sentiment(sample_text))
