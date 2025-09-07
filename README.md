# 🏠 House Prices: Advanced Regression Techniques

This is my implementation of the **Kaggle - House Prices: Advanced Regression Techniques** competition.  
The goal of the project is to predict house sale prices based on features such as lot area, number of rooms, building quality, and more.

---

# 📂 Quick Access
- 🔗 [Competition Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)  
- 🔗 [Kaggle](https://www.kaggle.com/code/fatimsadixova/house-price-prediction-random-forest-model)  
- 📒 [Notebooks in this Repository](notebooks/)

---

## 📌 Objective
My main goals in this project were:  
- To solve a real-world regression problem  
- To apply machine learning models for house price prediction  
- To design a clean and reproducible **Pipeline workflow** with preprocessing + modeling steps   

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
   - Local Validation RMSE: **0.14424**  
   - Kaggle Public Leaderboard Score: *(to be added here once available)*  

---

## 📈 Results
The pipeline-based model successfully predicted house prices with good accuracy.  

- **Validation RMSE:** 0.14424  
- Predictions were successfully submitted to Kaggle.  

---

## 🚀 Future Improvements
In the future, I plan to:  
- Experiment with more advanced models (XGBoost, LightGBM, Neural Networks)  
- Perform deeper feature engineering  
- Add model interpretability (SHAP values, feature importance)  

---

## 📂 Project Structure
