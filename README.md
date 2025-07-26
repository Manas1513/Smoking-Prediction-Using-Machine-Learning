# Smoking-Prediction-Using-Machine-Learning

# 🚬 Smoking Prediction Using Machine Learning

This repository contains a machine learning project that predicts whether an individual is a smoker or not based on various health-related attributes. It includes data preprocessing, model training, evaluation, and testing using Python.

## 📁 Project Structure

📦 Smoking-Prediction-ML
├── Smoking.py             # Main Python script for training and testing the model
├── smoking.csv            # Original dataset
├── x_train.csv            # Training input data
├── y_train.csv            # Training target labels
├── x_test.csv             # Testing input data
├── y_test.csv             # Testing target labels
└── README.md              # Project documentation

## 📌 Objective

The goal of this project is to classify whether a person is a **smoker or non-smoker** using health and demographic data. This can help in early intervention and health risk assessments.


## 📊 Dataset

* **Source**: The dataset `smoking.csv` contains various features like age, BMI, blood pressure, cholesterol, etc., along with the target column `smoking` (1 = Smoker, 0 = Non-smoker).
* **Shape**: The dataset is split into training and testing sets:

  * `x_train.csv` / `y_train.csv`
  * `x_test.csv` / `y_test.csv`

---

## 🧠 ML Approach

The `Smoking.py` script performs the following:

1. Loads the dataset
2. Preprocesses the data (scaling, splitting)
3. Trains a machine learning model (e.g., Logistic Regression / Random Forest / etc.)
4. Evaluates the model on test data
5. Prints accuracy and other metrics

---

## 🛠️ How to Run

### 🔧 Requirements

Install the dependencies using:

pip install -r requirements.txt


> **Note:** Create a `requirements.txt` using:
>
> pip freeze > requirements.txt

### 🚀 Run the model

python Smoking.py


Make sure all CSV files are in the same directory as the script.


## 📈 Example Output

Training Accuracy: 92.3%
Testing Accuracy: 89.7%
Classification Report:
Precision | Recall | F1-Score


## 📌 Future Improvements

* Add more ML models for comparison
* Use feature importance analysis
* Deploy as a web app using Streamlit or Flask


## 👨‍💻 Author

**Manas Tripathi**
Babu Banarasi Das University
LinkedIn - https://www.linkedin.com/in/manas--tripathi/

## 📄 License

This project is open-source and free to use.
