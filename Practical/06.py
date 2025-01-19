import numpy as np  # Importing a library for math functions like exponential

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example calculations
print(sigmoid(2))  # Output: 0.8808 (close to 1, meaning high probability of "Yes")
print(sigmoid(-0.9))  # Output: 0.2890 (close to 0, meaning low probability of "Yes")



#  logistic regression model using Scikit-Learn:

from sklearn.linear_model import LogisticRegression  # Import the model
from sklearn.model_selection import train_test_split  # To split data
from sklearn.datasets import load_iris  # Example dataset

# Load data (for example, Iris dataset)
data = load_iris()
X = data.data[:100, :2]  # Take 2 features (for simplicity)
y = data.target[:100]  # Binary target (0 or 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)  # Train the model

# Test the model
accuracy = model.score(X_test, y_test)  # Calculate accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
