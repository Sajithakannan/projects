import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Assuming y_test and y_pred are the true and predicted values, respectively
y_true = y_test  # Assuming y_test contains the true values of the target variable
y_pred = linear_reg.predict(X_test)  # Assuming linear_reg is your trained linear regression model

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Calculate R-squared
r2 = r2_score(y_true, y_pred)
print(f"R-squared: {r2:.4f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")

# Adjusted R^2 is not directly available in scikit-learn, but you can calculate it manually as shown in the previous code snippet

# Calculate Explained Variance Score
explained_variance = explained_variance_score(y_true, y_pred)
print(f"Explained variance score: {explained_variance:.4f}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"MAPE: {mape:.4f}")
