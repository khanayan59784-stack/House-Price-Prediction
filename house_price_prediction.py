# House Price Prediction using Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
data = pd.read_csv("house_data.csv")

print("\nDataset:")
print(data)

# 2. Explore data
print("\nFirst Five Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())


# 3. Define features and target
X = data[
    ["Area", "Bedrooms", "Bathrooms", "Age", "Parking", "Location"]
]
y = data["Price"]


# 4. Define categorical and numerical features
categorical_features = ["Location"]
numeric_features = [
    "Area", "Bedrooms", "Bathrooms", "Age", "Parking"
]


# 5. Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 6. Create ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# 7. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. Train model
model.fit(X_train, y_train)
print("\nModel Training Completed Successfully!")


# 9. Make predictions
predictions = model.predict(X_test)


# 10. Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# 11. Compare actual and predicted prices
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nActual vs Predicted:")
print(comparison)


# 12. Predict price for a new house
new_house = pd.DataFrame({
    "Area": [1600],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5],
    "Parking": [1],
    "Location": ["City"]
})

new_prediction = model.predict(new_house)

print("\n========== NEW HOUSE PREDICTION ==========")
print("New House Details:")
print(new_house)
print("\nPredicted House Price:", new_prediction[0])


# 13. Visualize actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, label="Predicted Prices")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()


# 14. Save trained model
joblib.dump(model, "model.pkl")
print("\nModel Saved Successfully as model.pkl")
