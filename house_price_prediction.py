# ============================================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ============================================================

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Load the house price dataset from CSV file
data = pd.read_csv("house_data.csv")

print("\nDataset:")
print(data)


# ============================================================
# 2. BASIC DATA EXPLORATION
# ============================================================

# Display first five rows
print("\nFirst Five Rows:")
print(data.head())

# Display number of rows and columns
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumn Names:")
print(data.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

# Features are the columns used to predict house price
X = data[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "Parking",
        "Location"
    ]
]

# Price is the target variable
y = data["Price"]


# ============================================================
# 4. IDENTIFY CATEGORICAL FEATURES
# ============================================================

# Location contains text values such as City and Suburb.
# OneHotEncoder converts these categories into numerical values.

categorical_features = ["Location"]

numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "Parking"
]


# ============================================================
# 5. PREPROCESS THE DATA
# ============================================================

# Convert categorical Location values into numerical values.
# Numeric columns are passed without changing them.

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


# ============================================================
# 6. CREATE MACHINE LEARNING PIPELINE
# ============================================================

# Pipeline first preprocesses the data and then trains
# the Linear Regression model.

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

# 80% data is used for training.
# 20% data is used for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

# Train the Linear Regression model using training data.

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

# Predict prices for the test dataset.

predictions = model.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

# MAE tells the average absolute prediction error.
mae = mean_absolute_error(y_test, predictions)

# MSE calculates the average squared error.
mse = mean_squared_error(y_test, predictions)

# RMSE is the square root of MSE.
rmse = mse ** 0.5

# R2 score shows how well the model explains the target values.
r2 = r2_score(y_test, predictions)


print("\n================ MODEL RESULTS ================")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# ============================================================
# 11. COMPARE ACTUAL AND PREDICTED VALUES
# ============================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nActual vs Predicted:")
print(comparison)


# ============================================================
# 12. PREDICT PRICE FOR A NEW HOUSE
# ============================================================

# Create data for a new house.
# Values:
# Area = 1600 sq ft
# Bedrooms = 3
# Bathrooms = 2
# Age = 5 years
# Parking = 1
# Location = City

new_house = pd.DataFrame({
    "Area": [1600],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5],
    "Parking": [1],
    "Location": ["City"]
})


# Predict the price of the new house
new_prediction = model.predict(new_house)


print("\n================ NEW HOUSE PREDICTION ================")
print("New House Details:")
print(new_house)

print("\nPredicted House Price:", new_prediction[0])


# ============================================================
# 13. VISUALIZE ACTUAL VS PREDICTED PRICES
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions,
    label="Predicted Prices"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 14. SAVE TRAINED MODEL
# ============================================================

# Save the complete pipeline.
# This allows us to use the trained model later
# without training it again.

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully as model.pkl")