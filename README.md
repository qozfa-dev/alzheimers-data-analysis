# alzheimers-data-analysis

This project uses Python and machine learning to explore and predict Alzheimer's disease outcomes based on clinical and demographic features.  
It includes data cleaning, preprocessing, feature scaling, and a Random Forest classifier for prediction and feature importance analysis.

## Steps Performed

1. **Data Cleaning:**  
   - Removed irrelevant or missing-value columns using `df.drop()`  
   - Checked dataset structure with `.info()` and summary statistics via `.describe()`

2. **Train-Test Split:**  
   - Used `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`  
   - Ensured stratified sampling to preserve class balance  

3. **Feature Scaling:**  
   - Applied `StandardScaler()` to standardize feature values  

4. **Model Building (Random Forest):**  
   - Created and trained a `RandomForestClassifier(random_state=42)`  
   - Predicted on test data and evaluated using:
     - Accuracy metrics  
     - ROC-AUC score  
     - Confusion matrix  
     - Classification report  

5. **Feature Importance Visualization:**  
   - Extracted feature importances using `rf.feature_importances_`  
   - Visualized the top 10 most important predictors with a horizontal bar chart  

---

## Key Results

- Random Forest achieved strong classification performance on test data  
- Identified top 10 most important features contributing to dementia prediction  
- Produced clear, interpretable visualizations for feature importance  

---

## How to Run the Project

1. **Clone this repository:**
   ```bash
   git clone https://github.com/qozfa-dev/alzheimers-data-analysis.git
   cd alzheimers-data-analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
3. **Run the notebook:**
   ```bash
   jupyter notebook alzheimers_data.ipynb
   ```

## Future Improvements
- Experiment with additional ML models (e.g., Logistic Regression, XGBoost)
- Add cross-validation for more robust performance metrics
- Include SHAP or LIME for explainable AI visualizations

