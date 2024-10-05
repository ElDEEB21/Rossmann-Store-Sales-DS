import pandas as pd
import joblib

# Load the trained SGDRegressor model
model = joblib.load('C:/Users/mosta/Downloads/model/SGDRegressor_best_model.pkl')

# Extract the weights (coefficients) and intercept
weights = model.coef_
intercept = model.intercept_

# Automatically generate feature names based on the number of weights
num_features = len(weights)
features = [f'Feature_{i+1}' for i in range(num_features)]  # Automatically generated feature names

# Check if the number of weights matches the number of features
if len(weights) != len(features):
    raise ValueError("The number of weights and features must be the same length!")

# Create a DataFrame to display weights
coefficients_df = pd.DataFrame({
    'Feature': features, 
    'Weight': weights
})

# Create a DataFrame for the intercept
intercept_df = pd.DataFrame({'Feature': ['Intercept'], 'Weight': intercept})

# Use pd.concat to combine weights and intercept
coefficients_df = pd.concat([coefficients_df, intercept_df], ignore_index=True)

# Display the DataFrame
result = coefficients_df
print(result.to_csv("weights.csv"),result)
