# data-science-analytics-internship-task2

# Credit Risk Prediction

## Objective
Predict whether a loan applicant is likely to default using machine learning models, and evaluate their performance.

## Dataset
- **Dataset:** German Credit Dataset (Kaggle)
- **Features:** Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose
- **Target:** Risk (`good` = 1, `bad` = 0)
- **Rows:** 1000

## Steps Performed
1. **Data Cleaning and Handling Missing Values**:
   - Filled numeric columns with median.
   - Filled categorical columns with mode.
2. **Visualization**:
   - Histogram of loan amount (`Credit amount`) to check distribution.
   - Bar plot of `Job` to examine category counts.
   - Histogram of `Credit amount / Duration` as income proxy.
3. **Feature Preparation**:
   - Target column mapped to binary.
   - One-hot encoding for categorical variables.
   - Split data into training and test sets (80/20) with stratification.
4. **Model Training**:
   - Logistic Regression (scaled features)
   - Decision Tree Classifier (max_depth=6)
5. **Evaluation**:
   - Accuracy, confusion matrix, classification report for both models.

## Key Insights
- Logistic Regression achieved ~69.5% accuracy; better at predicting `good` loans than `bad`.
- Decision Tree achieved ~70.5% accuracy; slightly better at capturing nonlinear interactions.
- Job, checking account, and credit amount play important roles in predicting default risk.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn
