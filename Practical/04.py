# simple Python code to create polynomial features using the sklearn library:

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Create polynomial features up to degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print(X_poly)
# Output: [[ 1.  1.  2.  1.  2.  4.]
#  [ 1.  3.  4.  9. 12. 16.]
#  [ 1.  5.  6. 25. 30. 36.]]


# Example Code for R² and Adjusted R²

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fit a linear regression model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_train)

# Calculate R²
r_squared = r2_score(y_train, y_pred)

# Adjusted R² formula
n = len(y_train)  # Number of samples
p = X_train.shape[1]  # Number of predictors
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("R²:", r_squared)
print("Adjusted R²:", adjusted_r_squared)
