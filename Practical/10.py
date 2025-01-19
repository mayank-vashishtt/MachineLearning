
import numpy as np  # For mathematical operations

def euclidean_distance(point1, point2):
    # Calculate the square root of the sum of squared differences
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Example
print(euclidean_distance(np.array([2, 3]), np.array([5, 7])))  # Output: 5.0



#How to Use KNN with scikit-learn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Example dataset
data = {
    'Weight': [150, 130, 160, 170, 180],
    'Size': [7, 6, 8, 8, 9],
    'Class': ['Apple', 'Apple', 'Orange', 'Orange', 'Orange']
}

# Convert it into a DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Weight', 'Size']]  # Features
y = df['Class']  # Target (Labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 neighbors

# Train the model
knn.fit(X_train, y_train)


# Predict for a new fruit
new_fruit = [[165, 8]]  # Weight = 165, Size = 8
prediction = knn.predict(new_fruit)
print("Prediction for the new fruit:", prediction[0])


# Make predictions for the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
