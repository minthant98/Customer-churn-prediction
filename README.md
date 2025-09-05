# Customer-churn-prediction
Customer Churn Prediction Web App

## Overview
This project predicts customer churn for a telecom company using machine learning. Users can input customer profile information, and the app predicts whether a customer is likely to churn, along with the probability.

The goal is to help businesses proactively retain customers and improve satisfaction.

## Features
- Handles categorical and numerical data with **label encoding** and optional **feature scaling**.
- Applies **SMOTE** to handle class imbalance for better minority class prediction.
- Uses a **Random Forest Classifier** with hyperparameter tuning for improved **F1-score**.
- Provides interactive **churn probability** with clear messages for end users.
- Built with **Streamlit** for an easy-to-use web interface.

## Technologies Used
- **Programming & ML:** Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Web Deployment:** Streamlit
- **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Model Storage:** Pickle
