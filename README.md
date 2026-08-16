# 🏠 House Price Prediction

### Machine Learning Regression Project

A Machine Learning project that predicts house prices using property features such as **area, bedrooms, bathrooms, property age, parking availability, and location**.

---

## 📌 About the Project

**House Price Prediction** is a supervised Machine Learning project developed using **Python and Scikit-learn**.

The project uses **Linear Regression** to learn the relationship between different house characteristics and their prices.

Unlike a simple numerical-only model, this project also handles the categorical **Location** feature using **One-Hot Encoding** and combines preprocessing with model training through a **Scikit-learn Pipeline**.

The complete workflow includes data loading, preprocessing, model training, prediction, evaluation, visualization, and model saving.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Load and analyze a house price dataset
* Identify input features and target values
* Handle categorical data
* Apply One-Hot Encoding to location
* Split data into training and testing sets
* Train a Linear Regression model
* Predict house prices
* Evaluate model performance
* Visualize actual vs predicted prices
* Save the trained Machine Learning model

---

## 🔄 Machine Learning Workflow

```text
                House Price Dataset
                        │
                        ▼
                 Data Loading
                        │
                        ▼
                Data Exploration
                        │
                        ▼
              Feature Selection
                        │
                        ▼
             Categorical Encoding
                  (Location)
                        │
                        ▼
                Train / Test Split
                        │
                        ▼
              Linear Regression
                        │
                        ▼
                 Model Training
                        │
                        ▼
                  Predictions
                        │
                        ▼
             Model Evaluation
                        │
                        ▼
                Data Visualization
                        │
                        ▼
                Save Model (.pkl)
```

---

## 🛠️ Technologies Used

| Technology      | Purpose                       |
| --------------- | ----------------------------- |
| 🐍 Python       | Programming language          |
| 🐼 Pandas       | Data loading and manipulation |
| 🔢 NumPy        | Numerical operations          |
| 📊 Matplotlib   | Data visualization            |
| 🤖 Scikit-learn | Machine Learning              |
| 💾 Joblib       | Saving the trained model      |

---

## 📊 Dataset

The project uses a custom house price dataset stored in:

```text
house_data.csv
```

### Features

| Feature     | Description                |
| ----------- | -------------------------- |
| `Area`      | Size of the house          |
| `Bedrooms`  | Number of bedrooms         |
| `Bathrooms` | Number of bathrooms        |
| `Age`       | Age of the property        |
| `Parking`   | Available parking spaces   |
| `Location`  | Property location category |
| `Price`     | Target house price         |

### Input Features

```text
Area
Bedrooms
Bathrooms
Age
Parking
Location
```

### Target

```text
Price
```

---

## 🧠 Machine Learning Model

### Linear Regression

The project uses **Linear Regression**, a supervised Machine Learning algorithm used for predicting continuous numerical values.

The model learns the relationship between house features and the target price.

Conceptually:

```text
House Features
      ↓
Machine Learning Model
      ↓
Predicted House Price
```

---

## 🔤 Categorical Feature Handling

The dataset contains a categorical feature:

```text
Location
```

For example:

```text
City
Suburb
```

Since Machine Learning models work with numerical values, the project converts these categories using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This preprocessing step is integrated with the Linear Regression model using a:

```python
Pipeline
```

This makes the training and prediction process consistent.

---

## 🧪 Training & Testing

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to learn patterns, while the testing data is used to evaluate the model on data it has not seen during training.

---

## 📈 Model Evaluation

The project evaluates the predictions using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

### MSE — Mean Squared Error

Measures the average squared prediction error.

### RMSE — Root Mean Squared Error

The square root of MSE and represents prediction error in the same scale as the target.

### R² Score

Measures how well the model explains the variation in house prices.

---

## 📊 Visualization

The project generates an:

### Actual vs Predicted House Prices

The graph compares the actual prices from the test dataset with the prices predicted by the Machine Learning model.

This provides a simple visual way to understand model performance.

---

## 🔮 New House Prediction

The trained model can also estimate the price of a new property.

Example:

```text
Area       : 1600
Bedrooms   : 3
Bathrooms  : 2
Age        : 5
Parking    : 1
Location   : City
```

These values are passed to the trained pipeline, which returns an estimated house price.

---

## 💾 Model Saving

After successful training, the complete trained pipeline is saved using Joblib:

```python
joblib.dump(model, "model.pkl")
```

This creates:

```text
model.pkl
```

The saved model can be reused later without training the model again.

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── 📄 house_price_prediction.py
├── 📊 house_data.csv
├── 🤖 model.pkl
├── 📦 requirements.txt
├── 📖 README.md
└── ⚙️ .gitignore
```

### File Details

**`house_price_prediction.py`**
Main Python program containing the complete Machine Learning workflow.

**`house_data.csv`**
Dataset containing house features and prices.

**`model.pkl`**
Saved trained Machine Learning pipeline.

**`requirements.txt`**
List of Python libraries required to run the project.

**`README.md`**
Project documentation.

**`.gitignore`**
Prevents unnecessary files from being tracked by Git.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/khanayan59784-stack/House-Price-Prediction.git
```

### 2. Open the Project Folder

```bash
cd House-Price-Prediction
```

### 3. Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run:

```bash
python house_price_prediction.py
```

The program will:

```text
✓ Load the dataset
✓ Analyze the data
✓ Check missing values
✓ Prepare features
✓ Encode Location
✓ Split the dataset
✓ Train the model
✓ Generate predictions
✓ Calculate evaluation metrics
✓ Display the visualization
✓ Save model.pkl
```

---

## 📌 Example Program Output

```text
Model Training Completed Successfully!

Mean Absolute Error: ...
Mean Squared Error: ...
Root Mean Squared Error: ...
R2 Score: ...

Predicted House Price: ...

Model Saved Successfully as model.pkl
```

> The actual metric values depend on the dataset and model training results.

---

## 🧠 Key Learning Outcomes

Through this project, I gained practical experience in:

* Python programming
* Data preprocessing
* Dataset exploration
* Feature selection
* Categorical data encoding
* One-Hot Encoding
* Train/Test splitting
* Linear Regression
* Machine Learning Pipelines
* Model evaluation
* Data visualization
* Model persistence
* Making predictions with trained models

---

## 🚀 Future Enhancements

The project can be extended by:

* Using a larger real-world dataset
* Adding more property features
* Comparing multiple regression algorithms
* Applying hyperparameter tuning
* Improving prediction accuracy
* Creating an interactive web interface
* Developing an API for predictions
* Deploying the model online

---

## 📌 Project Status

### ✅ Completed

This project demonstrates a complete beginner-level Machine Learning regression workflow, from dataset preparation to model training, evaluation, visualization, prediction, and model saving.

---

## 👨‍💻 Author

### PATHAN AYAN ASIF

**Machine Learning | Python Programming**

This project was developed as part of my practical Machine Learning learning and internship work.

---
## 🙏 Acknowledgement

Thank you to **SYNTECXHUB** for providing me with this internship opportunity and a platform to gain practical experience in Machine Learning and Python.

I am grateful for the guidance and learning experience throughout this project.

---
**Built with Python 🐍 and Machine Learning 🤖**
