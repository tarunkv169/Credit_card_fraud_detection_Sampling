
# Credit Card Fraud Detection

This project demonstrates various sampling techniques applied to a highly imbalanced dataset for credit card fraud detection. The dataset is used to predict fraudulent transactions, where the target variable (Class) indicates whether a transaction is fraud (1) or not fraud (0). The dataset consists of 772 rows and 31 columns, including features like time, principal components (V1 to V28), amount, and the target variable.

## Dataset Overview

- **Rows**: 772
- **Columns**: 31
- **Features**: Time, V1 to V28 (PCA components), Amount
- **Target variable**: Class (0: non-fraud, 1: fraud)
- **Missing Values**: None

### Class Distribution

- **Class 0 (Non-Fraud)**: 763 samples
- **Class 1 (Fraud)**: 9 samples

## Sampling Techniques Applied

1. **Original Data**
    - The initial dataset with imbalanced class distribution. The majority of the transactions are non-fraudulent, leading to biased model predictions.

2. **Undersampling**
    - The majority class (Class 0) is reduced to balance with the minority class (Class 1).
    - Outcome: Model struggles to detect fraud due to the reduced number of samples for Class 1.

3. **Oversampling**
    - The minority class (Class 1) is increased by duplicating samples to match the majority class (Class 0).
    - Outcome: The model shows perfect classification performance for both classes, as the dataset becomes balanced.

4. **SMOTE (Synthetic Minority Over-sampling Technique)**
    - Synthetic samples are generated for the minority class to balance the dataset.
    - Outcome: The model performs nearly perfectly for both classes, as the synthetic samples help improve the representation of Class 1.

5. **Cluster Sampling**
    - The dataset is divided into clusters, and samples are selected from these clusters.
    - Outcome: The model performs perfectly for Class 0 but struggles to detect fraud due to a lack of samples for Class 1 in the test set.

## Models Evaluated

The following machine learning models are evaluated to assess their performance on different sampling techniques:

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- SVM (Support Vector Machine)
- K-Nearest Neighbors (KNN)

## Results

- **Original Data**: The model performs well for Class 0 but fails to detect Class 1 due to the imbalanced data.
- **Undersampled Data**: Performance drops significantly, with a recall of 0.00 for Class 1.
- **Oversampled Data**: The model performs perfectly with a recall of 1.00 for both classes.
- **SMOTE Data**: The model performs nearly perfectly, with precision, recall, and F1-score close to 1.00 for both classes.
- **Cluster-Sampled Data**: The model detects Class 0 perfectly but fails to detect Class 1 due to insufficient samples.

## Conclusion

- **SMOTE** is the best option in this case for addressing the imbalance between the two classes. It generates synthetic samples for the minority class, improving the model's ability to detect fraudulent transactions without overfitting or losing valuable information from the majority class.
- **Oversampling** also shows good results, but it may lead to overfitting as it simply duplicates minority class samples.
- **Undersampling** leads to significant performance loss due to the reduction in samples, especially for the majority class.
- **Cluster Sampling** was not as effective because it didn’t provide enough instances of Class 1 for proper model training.

## Libraries Used

- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms and metrics
- `imbalanced-learn`: Sampling techniques (SMOTE, undersampling, oversampling)

## Installation

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## Usage

Run the notebook `Credit_card_fraud_detection(1).ipynb` step by step to reproduce the results.

## License

This project is licensed under the MIT License.
```
