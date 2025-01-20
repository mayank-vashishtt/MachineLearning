#Let’s use Python and Scikit-learn to implement AdaBoost.

# Importing necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a sample dataset
# Imagine we are predicting if a person has heart disease based on some features
X, y = make_classification(
    n_samples=500,     # Number of samples
    n_features=5,      # Number of features (like age, weight, etc.)
    n_classes=2,       # Binary classification (0 = No disease, 1 = Disease)
    random_state=42
)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize a weak learner (simple decision stump)
base_model = DecisionTreeClassifier(max_depth=1)  # A weak learner with only 1 split

# Step 4: Initialize AdaBoost
ada_model = AdaBoostClassifier(
    base_estimator=base_model,  # Weak learner
    n_estimators=50,            # Number of weak learners to combine
    learning_rate=1.0,          # Controls the contribution of each weak learner
    random_state=42
)

# Step 5: Train the AdaBoost model
ada_model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = ada_model.predict(X_test)

# Step 7: Check the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of AdaBoost model: {accuracy * 100:.2f}%")
