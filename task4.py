import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =======================
# Load Excel file
# =======================
file_path = r"C:\Users\hemal\OneDrive\Documents\task4.xlsx"
df = pd.read_excel(file_path)

# Check original columns
print("Original Columns:", df.columns)

# Select only the first 2 columns (adjust indexes if needed)
df = df.iloc[:, [0, 1]]  # taking first two columns
df.columns = ['Category', 'Message']

# =======================
# Data Cleaning
# =======================
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['Message'] = df['Message'].apply(clean_text)

# =======================
# Train-Test Split
# =======================
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =======================
# TF-IDF Vectorization
# =======================
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =======================
# Model Training
# =======================
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# =======================
# Evaluation
# =======================
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =======================
# 📊 Graph 1: Spam vs Ham count
# =======================
plt.figure(figsize=(6,4))
df['Category'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Spam vs Ham Email Count')
plt.xticks(ticks=[0,1], labels=['Ham (0)', 'Spam (1)'], rotation=0)
plt.ylabel('Count')
plt.show()

# =======================
# 📊 Graph 2: Accuracy Bar
# =======================
plt.figure(figsize=(4,4))
plt.bar(['Model Accuracy'], [acc], color='blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Spam Detection Model Accuracy')
plt.show()

# =======================
# Test with new emails
# =======================
test_emails = [
    "Congratulations! You won a lottery worth $1,000,000. Click here to claim.",
    "Dear user, your appointment is scheduled tomorrow at 3 PM.",
    "Hurry! Exclusive offer just for you. Limited time deal!!"
]

test_emails_clean = [clean_text(email) for email in test_emails]
test_emails_vec = vectorizer.transform(test_emails_clean)
predictions = model.predict(test_emails_vec)

print("\n📬 New Email Predictions:")
for email, label in zip(test_emails, predictions):
    print(f"Email: {email}\n → Predicted: {'SPAM' if label == 1 else 'NOT SPAM'}\n")
