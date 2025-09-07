# 🏠 House Prices: Advanced Regression Techniques

This is my implementation of the **Kaggle - House Prices: Advanced Regression Techniques** competition.  
The goal of the project is to predict house sale prices based on features such as lot area, number of rooms, building quality, and more.

---

# 📂 Quick Access
- 🔗 [Notebooks in this Repository](https://github.com/fatimasadikhova/House-Price-Prediction-Random-Forest-Model/blob/main/House%20Price%20Prediction-%20Random%20Forest%20Model.ipynb)
- 🔗 [Train Dataset](https://github.com/fatimasadikhova/House-Price-Prediction-Random-Forest-Model/blob/main/train.csv)
- 🔗 [Test Dataset](https://github.com/fatimasadikhova/House-Price-Prediction-Random-Forest-Model/blob/main/test.csv)
- 🔗 [Submission Dataset](https://github.com/fatimasadikhova/House-Price-Prediction-Random-Forest-Model/blob/main/submission.csv)
- 🔗 [Kaggle](https://www.kaggle.com/code/fatimsadixova/house-price-prediction-random-forest-model)  
---
## 📌 Objective
My main goals in this project were:  
- To solve a real-world regression problem  
- To apply machine learning models for house price prediction  
- To design a clean and reproducible **Pipeline workflow** with preprocessing + modeling steps  
- To optimize the model using **GridSearchCV**  

---

## 📊 Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Scikit-learn** (Pipeline, RandomForestRegressor, GridSearchCV)  
- **Jupyter Notebook** for experimentation  
- **Kaggle** for dataset and submission  

---

## 🔄 Project Workflow
I built the entire workflow using a **Pipeline** in scikit-learn, which allowed me to combine preprocessing and modeling into a single reproducible process:  

1. **Data Preparation**
   - Handled missing values  
   - Separated categorical and numerical features  
   - Checked and treated outliers  

2. **Feature Engineering (via Pipeline)**
   - OneHotEncoding for categorical features  
   - Scaling and transformations for numerical features  

3. **Modeling (via Pipeline)**
   - Random Forest Regressor  
   - Hyperparameter tuning with GridSearchCV  
   - Evaluation using RMSE  

4. **Evaluation**
   - Local Validation RMSE: **0.1447**  
   - Kaggle Public Leaderboard Score: **0.14421** 

---

## 🏗️ System Design
I also designed the **system architecture** for this project to demonstrate how the solution can be structured in a real-world scenario.  

**Key components of the design:**  
- **Data Layer:** Kaggle dataset, cleaned and preprocessed  
- **Processing Layer:** Data transformation & feature engineering using Pipelines  
- **Modeling Layer:** Random Forest Regressor with hyperparameter tuning  
- **Evaluation Layer:** Validation using RMSE and submission to Kaggle  
- **Deployment Idea (Future):** Model could be served as an API for real estate price prediction  

📌 [System Design](https://github.com/fatimasadikhova/House-Price-Prediction-Random-Forest-Model/blob/main/System%20Design.drawio.png)

---

## 📈 Results
The pipeline-based model successfully predicted house prices with good accuracy.  

- **RMSE:** 0.14424  
- Predictions were successfully submitted to Kaggle.  

---

## 🚀 Future Improvements
In the future, I plan to:  
- Experiment with more advanced models (XGBoost, LightGBM, Neural Networks)  
- Perform deeper feature engineering  
- Add model interpretability (SHAP values, feature importance)  
- Deploy the model as an API service  

---

