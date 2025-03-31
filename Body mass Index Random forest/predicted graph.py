import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Assuming your dataset is loaded into a DataFrame named 'df'
df = pd.read_csv('/content/bmi123.csv')

# Check the columns in your dataset
print(df.columns)

# Extracting features and target variable
X = df[['Age', 'Weight', 'Bmi']]
y = df['Height']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regressor
RFregressor = RandomForestRegressor()

# Train the regressor on the training data
RFregressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = RFregressor.predict(X_test)

# Compute residuals
residuals = y_test - y_pred

# Plot the residual graph
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Heights')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)
plt.show()
