import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# New Dataset
# Study Hours and Exam Marks
# -----------------------------

study_hours = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    2, 4, 6, 8, 10
]).reshape(-1, 1)

marks = np.array([
    35, 40, 45, 50, 55,
    60, 65, 70, 78, 85,
    42, 52, 63, 73, 88
])

print("X shape:", study_hours.shape)
print("y shape:", marks.shape)

# -----------------------------
# Split Data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    study_hours,
    marks,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Create and Train Model
# -----------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Model Results
# -----------------------------

print("\nModel Results")
print("--------------------")

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

print("MSE:", mean_squared_error(y_test, y_pred))

print("R² Score:", r2_score(y_test, y_pred))

# -----------------------------
# Example Prediction
# -----------------------------

new_hours = np.array([[7]])

prediction = model.predict(new_hours)

print("\nPrediction")
print("--------------------")
print("Study Hours:", new_hours[0][0])
print("Predicted Marks:", prediction[0])

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    study_hours,
    marks,
    marker="o",
    label="Actual Data"
)

plt.plot(
    study_hours,
    model.predict(study_hours),
    label="Regression Line"
)

plt.title("Study Hours vs Exam Marks")
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")

plt.legend()
plt.grid(True)

plt.show()