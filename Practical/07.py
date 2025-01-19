import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


# Example dataset
data = {
    'email_length': [50, 200, 30, 150, 300, 100, 20, 90, 400, 10],
    'num_special_chars': [5, 20, 3, 15, 25, 10, 1, 8, 30, 2],
    'is_spam': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['email_length', 'num_special_chars']]  # Features
y = df['is_spam']  # Labels


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
