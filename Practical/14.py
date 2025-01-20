#Let's look at the Python code for implementing Gradient Boosting using the sklearn library.

from sklearn.ensemble import GradientBoostingRegressor

# Sample data: features and target
X = [[1.6, 'blue', 'male'],
     [1.61, 'blue', 'female'],
     [0.51, 'green', 'female'],
     [0.81, 'blue', 'male'],
     [0.50, 'red', 'male'],
     [1.41, 'green', 'female'],
     [0.56, 'blue', 'female']]

y = [88, 76, 56, 73, 77, 57]  # Target variable (weight)

# Initialize and fit the Gradient Boosting model
model = GradientBoostingRegressor()  # This creates the Gradient Boosting Regressor model
model.fit(X, y)  # Train the model using our data

# Making predictions on new data
new_samples = [[1.65, 'blue', 'male'], [0.75, 'red', 'female']]  # New sample data
predictions = model.predict(new_samples)  # This will give the predicted values for these samples

print(predictions)
