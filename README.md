# Bank-Customer-Churn-Prediction
# Churn Prediction using Machine Learning
## Overview
This project implements multiple machine learning models to predict customer churn. The goal is to accurately classify customers as likely to churn or stay based on historical data. The following models have been evaluated and compared:

### Logistic Regression
### Support Vector Machine (SVM)
### K-Nearest Neighbors (KNN)
## Perceptron Model
The models' performances are assessed using metrics like accuracy, precision, recall, F1-score, and confusion matrices. Additionally, bootstrapping is used to estimate the stability and robustness of each model's performance.

## Models Evaluated
### 1. Logistic Regression
Accuracy: 82%
Precision (Class 0): 0.93
Recall (Class 0): 0.43
Precision (Class 1): 0.27
Recall (Class 1): 0.87
Macro avg F1: 0.60
Weighted avg F1: 0.78
Interpretation: Logistic regression performs well for class 0 but struggles with class 1, showing high recall but low precision for class 1.

### 2. Support Vector Machine (SVM)
Accuracy: 80%
Precision (Class 0): 0.80
Recall (Class 0): 1.00
Precision (Class 1): 0.00
Recall (Class 1): 0.00
Macro avg F1: 0.45
Weighted avg F1: 0.72
Interpretation: SVM performs perfectly for class 0 but fails to recognize any instances of class 1. Class imbalance severely impacts its performance.

### 3. K-Nearest Neighbors (KNN)
Accuracy: 82%
Precision (Class 0): 0.86
Recall (Class 0): 0.93
Precision (Class 1): 0.56
Recall (Class 1): 0.37
Macro avg F1: 0.67
Weighted avg F1: 0.80
Interpretation: The KNN model shows good performance for class 0 but has lower performance for class 1. Its overall accuracy is 82%, with a more balanced performance compared to logistic regression and SVM.

### 4. Perceptron Model
Average accuracy: 75%
Standard deviation: 0.10
Confidence Interval: 42.9% to 82.0%
Interpretation: The perceptron model has a wide confidence interval, indicating variability in performance across different iterations. While it shows an average accuracy of 75%, the model's performance fluctuates more than other models.

## Bootstrapping Results & Confidence Intervals
The bootstrapping method was used to estimate the variability and confidence intervals of model accuracies. The results are as follows:

### 1. Logistic Regression
Confidence Interval: 81.1% to 81.7%
Mean Precision: 0.81
Standard Deviation: 0.00
### 2. SVM
Confidence Interval: 80.3% to 80.3%
Mean Accuracy: 0.80
Standard Deviation: 0.00
### 3. KNN
Confidence Interval: 77.7% to 80.2%
Mean Precision: 0.79
Standard Deviation: 0.01
### 4. Perceptron
Confidence Interval: 42.9% to 82.0%
Mean Accuracy: 0.75
Standard Deviation: 0.10
## Key Insights
### Best Model: The SVM model performed best overall, achieving perfect recall for class 0 and high accuracy, but it failed to classify any instances of class 1.
Balanced Models: KNN and Logistic Regression showed more balanced performance, with KNN having slightly better overall accuracy and precision for class 0.
Confidence Intervals: Logistic regression and KNN classifiers show narrow confidence intervals, indicating high stability in their performance estimates. SVM's confidence interval is narrow as well, though it reflects its inability to classify class 1 instances effectively.
## Improvements
Handling Class Imbalance: Implement techniques such as oversampling, undersampling, or SMOTE to balance the classes and improve model performance for class 1.
### Model Ensemble: Consider using ensemble methods like bagging or boosting to combine the strengths of various models.
### Hyperparameter Tuning: Further refine the models through hyperparameter optimization to improve classification performance for both classes.
### Feature Engineering: Investigate the impact of additional features to enhance model predictions, particularly for the minority class (class 1).
## Conclusion
This project demonstrates how different machine learning algorithms can be applied to predict customer churn. While the SVM model performs well for class 0, logistic regression and KNN provide a more balanced solution. Future improvements in handling class imbalance and further fine-tuning the models can help achieve better overall accuracy.

Next Steps:
Implement the suggested improvements to further optimize the models.
Test the models on additional datasets to evaluate their generalization capability.
Deploy the best-performing model in a production environment.
