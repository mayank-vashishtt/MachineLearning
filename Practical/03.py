#Code for Linear Regression

# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
# Replace 'path_to_file.csv' with the path to your CSV file.
df = pd.read_csv('path_to_file.csv')

# Step 2: Explore the dataset
print("First 5 rows of the dataset:")
print(df.head())  # Display the first few rows of the dataset

print("\nColumns in the dataset:")
print(df.columns)  # Display column names

# Step 3: Handle missing values (if any)
print("\nChecking for missing values:")
print(df.isnull().sum())  # Check for missing values in each column
df = df.dropna()  # Drop rows with missing values

# Step 4: Encode categorical variables
# Example: Encoding the 'Make' column using mean encoding
if 'Make' in df.columns:
    df['Make'] = df['Make'].map(df.groupby('Make')['Selling Price'].mean())

# Step 5: Define features (X) and target variable (Y)
X = df.drop('Selling Price', axis=1)  # Features (all columns except target)
y = df['Selling Price']  # Target variable

# Step 6: Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² Score

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Step 11: Display the model's coefficients
print("\nModel Coefficients:")
print(f"Intercept (B): {model.intercept_}")
print(f"Weights (W): {model.coef_}")
