import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic data (quadratic function with noise)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Random X values (0 to 10)
y = 2 * (X**2) - 3 * X + 4 + np.random.randn(100, 1) * 10  # Quadratic with noise

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Helper function to fit and plot models
def fit_and_plot(degree, model, title):
    pipeline = make_pipeline(PolynomialFeatures(degree), model)
    pipeline.fit(X_train, y_train)
    
    # Predictions
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)  # For smooth curve
    y_plot = pipeline.predict(X_plot)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Plot
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='green', label='Test Data')
    plt.plot(X_plot, y_plot, color='red', label=f'Model (degree={degree})')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    
    # Print performance
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    print(f"Train Error: {train_error:.2f}, Test Error: {test_error:.2f}")

# 3. Underfitting: Model with low complexity
print("Underfitting Example (degree=1)")
fit_and_plot(degree=1, model=LinearRegression(), title="Underfitting (degree=1)")

# 4. Overfitting: Model with high complexity
print("Overfitting Example (degree=15)")
fit_and_plot(degree=15, model=LinearRegression(), title="Overfitting (degree=15)")

# 5. Balanced Model: Model with medium complexity
print("Balanced Model Example (degree=2)")
fit_and_plot(degree=2, model=LinearRegression(), title="Balanced Model (degree=2)")

# 6. Regularization: L1 (Lasso) and L2 (Ridge)
print("Regularization with L1 (Lasso)")
fit_and_plot(degree=15, model=Lasso(alpha=0.1, max_iter=10000), title="L1 Regularization (Lasso, degree=15)")

print("Regularization with L2 (Ridge)")
fit_and_plot(degree=15, model=Ridge(alpha=1), title="L2 Regularization (Ridge, degree=15)")
