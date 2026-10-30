# Task-2-Apurv-gaurav

# Project 2: Fraud Detection Pipeline (Supervised Learning)

## Project Overview

This project builds and tunes machine learning classifiers to detect fraudulent orders in an e-commerce dataset (`week 2 project.csv`, 1,200 rows × 14 columns). Fraud is rare compared with legitimate orders, so the project focuses on handling **class imbalance** with **SMOTE** and on evaluating the models with **Precision, Recall, F1 and ROC-AUC** instead of Accuracy.

Accuracy is deliberately not used. When almost every order is legitimate, a model that predicts "Legit" every time scores a very high accuracy while catching zero fraud.

This project is part of the DecodeLabs Data Science Industrial Training Kit (Batch 2026).

## Files in this Repository

| File | Description |
|------|-------------|
| `week 2 project.py` | Full pipeline: feature engineering, SMOTE, model training, tuning, evaluation |
| `week 2 project.csv` | Input dataset (1,200 rows, 14 columns) |
| `cleaned_project2.csv` | Output dataset with engineered fraud features and the `IsFraud` label (1,200 rows, 20 columns) |
| `README.md` | Project documentation |

## Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| OrderID | String | Unique order identifier |
| Date | Date | Order date |
| CustomerID | String | Customer identifier |
| Product | String | Product ordered |
| Quantity | Integer | Units ordered |
| UnitPrice | Float | Price per unit |
| ShippingAddress | String | Delivery address |
| PaymentMethod | String | Payment type |
| OrderStatus | String | Cancelled, Delivered, Pending, Returned, Shipped |
| TrackingNumber | String | Shipment tracking ID |
| ItemsInCart | Integer | Number of items in the cart |
| CouponCode | String | Coupon used (if any) |
| ReferralSource | String | Email, Facebook, Google, Instagram, Referral |
| TotalPrice | Float | Total order price |

### Engineered Columns (added in `cleaned_project2.csv`)

| Column | Meaning |
|--------|---------|
| LowUnitPrice | 1 if `UnitPrice` < 50 |
| HighRiskPayment | 1 if payment method is Gift Card or Cash |
| SharedAddress | 1 if the shipping address appears in more than one order |
| CancelledOrReturned | 1 if the order is Cancelled/Returned **and** has a tracking number |
| FlagCount | Sum of the four flags above (0–4) |
| IsFraud | **Target label**: 1 if `FlagCount` ≥ 3, else 0 |

## Tasks Performed

1. Located and loaded the dataset, parsing the `Date` column
2. Filled missing coupon codes with `No Coupon`
3. Engineered four fraud-risk flags and combined them into `FlagCount`
4. Created the `IsFraud` target (rule: at least 3 of 4 flags raised)
5. Checked the class distribution (count and percentage)
6. Built the feature matrix (numeric, date-based and one-hot encoded features)
7. Split the data 80/20 with a stratified train/test split
8. Built two leak-free `imblearn` pipelines (Logistic Regression and Random Forest) with SMOTE inside
9. Tuned both models with `GridSearchCV` and 5-fold Stratified Cross-Validation, scoring on ROC-AUC
10. Evaluated both models on the untouched test set (confusion matrix, Precision, Recall, F1, ROC-AUC)
11. Compared the models and saved the engineered dataset to CSV

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pathlib` | File path handling that works across operating systems |
| `pandas` | Data loading, feature engineering, encoding |
| `scikit-learn` | Train/test split, scaling, models, cross-validation, metrics |
| `imbalanced-learn` | `SMOTE` and the resampling-aware `Pipeline` |

## Line-by-Line Code Explanation

### 1. Imports
```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
```
- `Path` builds file paths safely on any operating system.
- `pandas` handles the data tables.
- `train_test_split` splits the data, `GridSearchCV` tunes hyperparameters, and `StratifiedKFold` creates cross-validation folds that keep the class ratio.
- `StandardScaler` standardizes features (mean 0, standard deviation 1).
- `LogisticRegression` and `RandomForestClassifier` are the two classification models.
- The metrics imports are used to evaluate the models.
- `imblearn.pipeline.Pipeline` is used instead of `sklearn.pipeline.Pipeline` because it supports resampling steps like SMOTE, and applies them **only to training folds**, which prevents data leakage.
- `SMOTE` creates synthetic minority-class (fraud) samples.

### 2. Locating the CSV file
```python
file_name = 'week 2 project.csv'
folders = [
    Path(__file__).resolve().parent,
    Path.cwd(),
    Path.home() / 'Downloads',
    Path.home() / 'Downloads' / 'week 2 project.py',
    Path.home() / 'Desktop',
    Path.home() / 'Documents'
]
```
- Stores the CSV file name.
- Builds a list of folders to search: the script's own folder, the current working directory, Downloads, Desktop and Documents.

```python
csv_path = None
for folder in folders:
    if (folder / file_name).exists():
        csv_path = folder / file_name
        break
```
- Starts with no path, then checks each folder in order.
- The first folder that contains the file is used, and the loop stops.

```python
if csv_path is None:
    for match in (Path.home() / 'Downloads').rglob(file_name):
        csv_path = match
        break
```
- If the file wasn't found, searches **all subfolders** of Downloads recursively and uses the first match.

```python
if csv_path is None:
    raise FileNotFoundError('Could not find ' + file_name + '. Put it in your Downloads folder or next to this script.')
```
- If the file still wasn't found, stops with a clear error message.

### 3. Loading the data
```python
print('Using file:', csv_path)
df = pd.read_csv(csv_path, parse_dates=['Date'])
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
print(df.shape)
```
- Prints which file is being used.
- Reads the CSV and converts the `Date` column to a real datetime type.
- Fills missing coupon codes with `'No Coupon'`, since a blank means no coupon was used.
- Prints the shape (rows, columns).

### 4. Feature engineering: fraud flags
```python
df['LowUnitPrice'] = (df['UnitPrice'] < 50).astype(int)
```
- Flags orders with a unit price below 50 as 1, otherwise 0.

```python
df['HighRiskPayment'] = df['PaymentMethod'].isin(['Gift Card', 'Cash']).astype(int)
```
- Flags orders paid by Gift Card or Cash, which are harder to trace.

```python
df['SharedAddress'] = (df.groupby('ShippingAddress')['OrderID'].transform('count') > 1).astype(int)
```
- Counts how many orders go to each shipping address and flags addresses used more than once.

```python
df['CancelledOrReturned'] = (df['OrderStatus'].isin(['Cancelled', 'Returned']) & df['TrackingNumber'].notnull()).astype(int)
```
- Flags orders that are Cancelled or Returned **and** have a tracking number (the order shipped but still came back or was cancelled).

### 5. Creating the target label
```python
df['FlagCount'] = df['LowUnitPrice'] + df['HighRiskPayment'] + df['SharedAddress'] + df['CancelledOrReturned']
df['IsFraud'] = (df['FlagCount'] >= 3).astype(int)
```
- `FlagCount` adds up the four flags (0 to 4).
- `IsFraud` is 1 when at least 3 flags are raised, otherwise 0. This is the column the models learn to predict.

```python
print(df['IsFraud'].value_counts())
print(df['IsFraud'].value_counts(normalize=True).round(4) * 100)
```
- Prints how many fraud and legit orders there are, then the same as percentages, which shows how imbalanced the data is.

### 6. Building the feature matrix
```python
X = pd.DataFrame({
    'Quantity': df['Quantity'],
    'ItemsInCart': df['ItemsInCart'],
    'Month': df['Date'].dt.month,
    'DayOfWeek': df['Date'].dt.dayofweek
})
```
- Starts the feature table with two numeric columns, plus the month and the day of the week (0 = Monday) taken from `Date`.

```python
X = pd.concat([X, pd.get_dummies(df[['Product', 'CouponCode', 'ReferralSource']], drop_first=True)], axis=1).astype(float)
y = df['IsFraud']
```
- One-hot encodes `Product`, `CouponCode` and `ReferralSource` into 0/1 columns.
- `drop_first=True` drops one column per category to avoid redundant (perfectly correlated) columns.
- Joins everything together and converts all values to float so the models can use them.
- `y` is the target (`IsFraud`).

### 7. Train/test split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.sum(), y_test.sum())
```
- 80% of the data goes to training and 20% to testing.
- `stratify=y` keeps the same fraud ratio in both sets.
- `random_state=42` makes the split reproducible.
- Prints the shapes and the number of fraud cases in each set.

### 8. Building the pipelines
```python
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
```
- **Logistic Regression pipeline:** scale features, then apply SMOTE, then train the classifier.
- Scaling is needed because Logistic Regression is sensitive to feature scale.
- `max_iter=1000` gives the solver enough iterations to converge.

```python
rf_pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])
```
- **Random Forest pipeline:** SMOTE then the classifier, with no scaler because tree-based models don't depend on feature scale.
- `n_jobs=-1` uses all CPU cores.

### 9. Hyperparameter grids
```python
lr_params = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__C': [0.01, 0.1, 1.0]
}
```
- Tries 3 SMOTE neighbour settings and 3 regularization strengths (`C`; smaller means stronger regularization): **9 combinations**.

```python
rf_params = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__max_depth': [10, 20, None],
    'classifier__n_estimators': [100, 200]
}
```
- Tries 3 SMOTE neighbour settings, 3 tree depths and 2 forest sizes: **18 combinations**.
- The `step__parameter` naming tells the pipeline which step each parameter belongs to.

### 10. Cross-validation and model list
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = [('Logistic Regression', lr_pipe, lr_params), ('Random Forest', rf_pipe, rf_params)]
scores = []
```
- Sets up 5-fold stratified cross-validation, shuffled and reproducible.
- Groups each model's name, pipeline and grid so they can be run in a loop.
- `scores` is an empty list that collects the final metrics.

### 11. Training, tuning and evaluation loop
```python
for name, pipe, params in models:
    grid = GridSearchCV(pipe, params, scoring='roc_auc', cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
```
- For each model, creates a grid search that scores every parameter combination by **ROC-AUC**.
- `fit` trains all combinations. SMOTE runs **inside each fold on the training part only**, so the validation fold stays untouched and there is no leakage.

```python
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]
```
- Uses the best model found to predict labels on the test set.
- `y_prob` holds the predicted probability of fraud, which is needed for ROC-AUC.

```python
    print('\n' + name)
    print('Best params:', grid.best_params_)
    print('CV ROC-AUC:', round(grid.best_score_, 4))
    print(confusion_matrix(y_test, y_pred))
```
- Prints the model name, the best hyperparameters, the best cross-validated ROC-AUC, and the confusion matrix (TN, FP, FN, TP).

```python
    report = pd.DataFrame(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], output_dict=True, zero_division=0)).T
    print(report.loc[['Legit', 'Fraud'], ['precision', 'recall', 'f1-score', 'support']].round(3))
```
- Builds the classification report as a dictionary, converts it to a table, and prints Precision, Recall, F1 and support for the Legit and Fraud classes only.
- `zero_division=0` avoids errors when a class has no predictions.

```python
    scores.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
```
- Stores the fraud-class Precision, Recall, F1 and ROC-AUC for this model so both models can be compared.

### 12. Final comparison and saving
```python
print('\nFinal comparison')
print(pd.DataFrame(scores).round(3).to_string(index=False))
```
- Prints a side-by-side comparison table of both models.

```python
save_path = Path.home() / 'Downloads' / 'cleaned_project2.csv'
df.to_csv(save_path, index=False)
print('Cleaned file saved at:', save_path)
```
- Saves the dataset with all the engineered columns and the `IsFraud` label to `cleaned_project2.csv` (1,200 rows × 20 columns).

## Metrics Used (and Why)

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | When the model flags fraud, how often is it right? |
| Recall | TP / (TP + FN) | Of all real fraud, how much did the model catch? |
| F1-score | Harmonic mean of Precision and Recall | Balance between the two |
| ROC-AUC | Area under the ROC curve | How well the model separates fraud from legit |

## Why the Pipeline is Leak-Free

- The train/test split happens **before** any scaling or SMOTE.
- `StandardScaler` and `SMOTE` live **inside** the `imblearn` pipeline, so they are fitted on training folds only.
- The test set stays untouched and keeps its real class imbalance.

## Results

> Run the script and fill in your own numbers below.

**Class distribution:** Legit = `___` (`__`%), Fraud = `___` (`__`%)

| Model | Best Parameters | CV ROC-AUC | Precision | Recall | F1 | ROC-AUC (Test) |
|-------|-----------------|-----------|-----------|--------|----|----------------|
| Logistic Regression | | | | | | |
| Random Forest | | | | | | |

## How to Run

1. Install the dependencies:
   ```bash
   pip install pandas scikit-learn imbalanced-learn
   ```
2. Place `week 2 project.csv` in the same folder as the script (or in your Downloads folder).
3. Run:
   ```bash
   python "week 2 project.py"
   ```
4. The output file `cleaned_project2.csv` is saved in your Downloads folder.

## Limitations and Possible Improvements

- The `IsFraud` label is **rule-based** (a synthetic label built from four flags), not confirmed real-world fraud. The model is therefore learning to approximate those rules from other columns.
- The features used for training (`Quantity`, `ItemsInCart`, date parts, `Product`, `CouponCode`, `ReferralSource`) are different from the ones used to build the label. This avoids direct label leakage but may limit model performance.
- `SharedAddress` is computed on the full dataset before the split. Computing it on the training set only would be stricter.
- Try other models (e.g., XGBoost) and other resampling methods (e.g., ADASYN, SMOTE-Tomek).
- Tune the decision threshold to trade Precision against Recall.
- Plot ROC and Precision-Recall curves and feature importances.

## Author

**Apurv Gaurav**
