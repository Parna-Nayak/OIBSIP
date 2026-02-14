# ======================================
# SENTIMENT ANALYSIS - CLASSICAL ML
# ======================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Settings
sns.set(style="whitegrid")
plt.style.use("ggplot")

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# 2️⃣ Load Dataset
df = pd.read_csv("C:\OASIS\PROJECT4\Twitter_Data_Cleaned.csv")  # raw string for Windows path

# Drop missing values
df.dropna(inplace=True)

# Ensure 'category' column is int
df['category'] = df['category'].astype(int)

# Display basic info
print("Dataset Shape:", df.shape)
print(df.head())
print("\nLabel Distribution:\n", df['category'].value_counts())

# 3️⃣ Data Visualization - Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['category'])
plt.title("Sentiment Class Distribution")
plt.xticks([0,1,2],['Negative','Neutral','Positive'])
plt.ylabel("Count")
plt.show()

# 4️⃣ Text Cleaning Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()                    # lowercase
    text = re.sub(r'http\S+|www\S+', '', text) # remove URLs
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'[^a-z\s]', '', text)       # keep only letters
    text = text.strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['clean_text'].apply(clean_text)

# Optional: WordCloud of all text
all_words = ' '.join(df['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of Tweets")
plt.show()

# 5️⃣ Feature Engineering - TF-IDF
X = df['clean_text']
y = df['category']

tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(X)

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Logistic Regression Model
log_reg = LogisticRegression(max_iter=2000, n_jobs=-1)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# 8️⃣ Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# 9️⃣ Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14,5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_nb = confusion_matrix(y_test, y_pred_nb)
labels = ['Negative','Neutral','Positive']

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("Naive Bayes")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

#  🔟 Model Comparison
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_nb)]
})

print("\nModel Comparison:\n", comparison_df)
