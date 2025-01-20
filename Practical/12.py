#decision tree using gini

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X, y = data.data, data.target  # Features and labels

# Train Decision Tree
dt_classifier = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
dt_classifier.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

# Predictions
predictions = dt_classifier.predict(X[:5])
print("Predicted labels:", predictions)
print("True labels:", y[:5])


#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#bagging

from sklearn.ensemble import BaggingClassifier

# Bagging Classifier with Decision Tree
bagging_classifier = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bagging_classifier.fit(X_train, y_train)

# Predictions and Accuracy
bagging_predictions = bagging_classifier.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, bagging_predictions))

#boosting using adaboost

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost Classifier
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# Predictions
adaboost_predictions = adaboost.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, adaboost_predictions))

#Python Example: Grid Search

from sklearn.model_selection import GridSearchCV

# Parameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

# Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
