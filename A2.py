import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the dataset
try:
    data_path = 'spam.csv'  # Path to the dataset
    dataset = pd.read_csv(data_path, encoding='latin-1')
    dataset = dataset[['Category', 'Message']]  # Ensure only the required columns are used
except FileNotFoundError:
    print(f"Error: Could not find the file {data_path}")
    exit(1)
except Exception as e:
    print(f"Error loading the dataset: {str(e)}")
    exit(1)

# Encode the target variable ('Category') as binary
dataset['Category'] = dataset['Category'].map({'ham': 0, 'spam': 1})

# Clean the messages
dataset['Message'] = dataset['Message'].str.replace(r'[^\w\s]', '', regex=True).str.lower()

# Features and Target
X = dataset['Message']
y = dataset['Category']

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=None, max_features=5000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression(C=1.0, max_iter=500)
log_reg.fit(X_train, y_train)

# Train Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Evaluate models on the test set
y_pred_log = log_reg.predict(X_test)
y_pred_nb = nb.predict(X_test)

# Calculate accuracy and F1 score for Logistic Regression
log_accuracy = accuracy_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log, average='weighted')

# Calculate accuracy and F1 score for Naive Bayes
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')

# Display scores
print(f"Logistic Regression Accuracy: {log_accuracy * 100:.2f}%")
print(f"Logistic Regression F1 Score: {log_f1:.2f}")
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(f"Naive Bayes F1 Score: {nb_f1:.2f}")

# Bar graph comparison
labels = ["Logistic Regression", "Naive Bayes"]
accuracy_scores = [log_accuracy, nb_accuracy]
f1_scores = [log_f1, nb_f1]

x = range(len(labels))
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x, accuracy_scores, width=width, label="Accuracy", color="skyblue", align="center")
plt.bar([i + width for i in x], f1_scores, width=width, label="F1 Score", color="orange", align="center")

plt.xlabel("Models")
plt.ylabel("Scores")
plt.title("Comparison of Accuracy and F1 Scores")
plt.xticks([i + width / 2 for i in x], labels)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Main prediction loop
while True:
    print("\n=== Spam Message Predictor ===")
    print("Choose an option:")
    print("1. Enter a single message for prediction")
    print("2. Predict for multiple messages")
    print("3. Exit")
    choice = input("Enter your choice: ").strip()

    if choice == '1':
        # Single message prediction
        user_input = input("Enter a message to classify: ").strip()
        new_data_vectorized = vectorizer.transform([user_input])
        log_reg_pred = log_reg.predict(new_data_vectorized)[0]
        nb_pred = nb.predict(new_data_vectorized)[0]
        print(f"\nPrediction Results:")
        print(f"Message: {user_input}")
        print(f"Logistic Regression predicts: {'Spam' if log_reg_pred == 1 else 'Ham'}")
        print(f"Naive Bayes predicts: {'Spam' if nb_pred == 1 else 'Ham'}")

    elif choice == '2':
        # Multiple messages prediction
        print("\nEnter messages (type 'END' when done):")
        messages = []
        while True:
            msg = input().strip()
            if msg.upper() == 'END':
                break
            messages.append(msg)

        new_data_vectorized = vectorizer.transform(messages)
        log_reg_preds = log_reg.predict(new_data_vectorized)
        nb_preds = nb.predict(new_data_vectorized)

        print("\nPrediction Results for Multiple Messages:")
        for i, msg in enumerate(messages):
            print(f"Message: {msg}")
            print(f"  Logistic Regression predicts: {'Spam' if log_reg_preds[i] == 1 else 'Ham'}")
            print(f"  Naive Bayes predicts: {'Spam' if nb_preds[i] == 1 else 'Ham'}")

    elif choice == '3':
        print("Thank you for using the Spam Message Predictor!")
        break

    else:
        print("Invalid choice. Please try again.")
