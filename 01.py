# Python Code to Calculate MSE

# Actual and predicted values
actual = [3, -0.5, 2, 7]       # Real values (y)
predicted = [2.5, 0.0, 2.1, 8] # Predicted values (ŷ)

# Step 1: Calculate Errors
errors = [a - p for a, p in zip(actual, predicted)]

# Step 2: Square Each Error
squared_errors = [error**2 for error in errors]

# Step 3: Calculate the Mean of Squared Errors
mse = sum(squared_errors) / len(squared_errors)

# Print the result
print("Mean Squared Error (MSE): {mse}")


#-----------------------------------------------

#Visualizing MSE

import matplotlib.pyplot as plt

# Actual vs Predicted values
actual = [3, -0.5, 2, 7]
predicted = [2.5, 0.0, 2.1, 8]

# Plot the data points with labels
plt.scatter(range(len(actual)), actual, color='blue', label='Actual Values')
plt.scatter(range(len(predicted)), predicted, color='red', label='Predicted Values')

# Draw lines showing the errors
# Add the error line just once to the legend
plt.plot([0, 0], [actual[0], predicted[0]], 'k--', label='Error')
# Draw the rest of the error lines without adding to legend
for i in range(1, len(actual)):
    plt.plot([i, i], [actual[i], predicted[i]], 'k--')

# Add title and legend
plt.title('Actual vs Predicted Values with Errors')
plt.legend()  # Now this will work because we have proper labels
plt.show()



#-----------------------------------------------
# Here, we’ll use LinearRegression from Scikit-Learn, which internally uses Gradient Descent to optimize the model parameters.



from sklearn.linear_model import LinearRegression
import numpy as np

# Data: x (input), y (real output)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshape to 2D for sklearn
y = np.array([2.2, 4.0, 5.8, 8.4, 10.2])

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(x, y)

# Get the optimized parameters
W = model.coef_[0]  # Slope (Weight)
b = model.intercept_  # Intercept

# Make predictions
y_pred = model.predict(x)

# Print results
print(f"Slope (W): {W}")
print(f"Intercept (b): {b}")
print("Predictions:", y_pred)



#-----------------------------------------------
# Here’s a small program that helps you work with structured data (like a spreadsheet):
import pandas as pd

# Load structured data
data = pd.read_csv('data.csv')

# Basic data manipulation
print(data.head())  # Shows the first 5 rows of the data
print(data.describe())  # Gives summary statistics about the data
