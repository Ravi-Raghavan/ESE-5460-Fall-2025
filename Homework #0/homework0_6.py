## Ravi Raghavan, Homework #0, Problem 6

## Library Imports
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
ds = fetch_california_housing()

# Store results
training_errors = []
validation_errors = []

# Perform the experiment 3 times
for i in range(3):
    print(f"\n--- Experiment {i+1} ---")
    
    # Split the data into 80% for training and 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        ds.data, ds.target, test_size=0.2, random_state=42 + i
    )
    
    # Add a bias term (a column of ones) to the feature matrices
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    
    # Calculate the analytical solution using stuff from part (a)
    # hat_w = (X^T * X)^-1 * X^T * y
    try:
        XTX = X_train_b.T @ X_train_b
        XTX_inv = np.linalg.inv(XTX)
        XTy = X_train_b.T @ y_train
        theta_optimal = XTX_inv @ XTy
    ## Error Handling
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Cannot compute analytical solution.")
        continue

    b_optimal = theta_optimal[0]
    w_optimal = theta_optimal[1:]
    
    # Make predictions using the calculated weights and bias
    y_train_pred = X_train_b @ theta_optimal
    y_val_pred = X_val_b @ theta_optimal
    
    # Calculate the mean squared error for both sets
    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)
    
    # Store Results
    training_errors.append(train_error)
    validation_errors.append(val_error)
    
    # Print Results for this Experiment
    print(f"Training MSE: {train_error:.4f}")
    print(f"Validation MSE: {val_error:.4f}")

# Calculate and report the mean and standard deviation of the errors
print("\n--- Summary ---")
print(f"Mean Training Error: {np.mean(training_errors):.4f}")
print(f"Standard Deviation of Training Error: {np.std(training_errors):.4f}")
print(f"Mean Validation Error: {np.mean(validation_errors):.4f}")
print(f"Standard Deviation of Validation Error: {np.std(validation_errors):.4f}")