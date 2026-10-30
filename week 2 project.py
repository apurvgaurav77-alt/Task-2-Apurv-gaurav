
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

file_name = 'week 2 project.csv'
folders = [
    Path(__file__).resolve().parent,
    Path.cwd(),
    Path.home() / 'Downloads',
    Path.home() / 'Downloads' / 'week 2 project.py',
    Path.home() / 'Desktop',
    Path.home() / 'Documents'
]

csv_path = None
for folder in folders:
    if (folder / file_name).exists():
        csv_path = folder / file_name
        break

if csv_path is None:
    for match in (Path.home() / 'Downloads').rglob(file_name):
        csv_path = match
        break

if csv_path is None:
    raise FileNotFoundError('Could not find ' + file_name + '. Put it in your Downloads folder or next to this script.')

print('Using file:', csv_path)
df = pd.read_csv(csv_path, parse_dates=['Date'])
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
print(df.shape)

df['LowUnitPrice'] = (df['UnitPrice'] < 50).astype(int)
df['HighRiskPayment'] = df['PaymentMethod'].isin(['Gift Card', 'Cash']).astype(int)
df['SharedAddress'] = (df.groupby('ShippingAddress')['OrderID'].transform('count') > 1).astype(int)
df['CancelledOrReturned'] = (df['OrderStatus'].isin(['Cancelled', 'Returned']) & df['TrackingNumber'].notnull()).astype(int)

df['FlagCount'] = df['LowUnitPrice'] + df['HighRiskPayment'] + df['SharedAddress'] + df['CancelledOrReturned']
df['IsFraud'] = (df['FlagCount'] >= 3).astype(int)

print(df['IsFraud'].value_counts())
print(df['IsFraud'].value_counts(normalize=True).round(4) * 100)

X = pd.DataFrame({
    'Quantity': df['Quantity'],
    'ItemsInCart': df['ItemsInCart'],
    'Month': df['Date'].dt.month,
    'DayOfWeek': df['Date'].dt.dayofweek
})
X = pd.concat([X, pd.get_dummies(df[['Product', 'CouponCode', 'ReferralSource']], drop_first=True)], axis=1).astype(float)
y = df['IsFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.sum(), y_test.sum())

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

rf_pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

lr_params = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__C': [0.01, 0.1, 1.0]
}

rf_params = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__max_depth': [10, 20, None],
    'classifier__n_estimators': [100, 200]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = [('Logistic Regression', lr_pipe, lr_params), ('Random Forest', rf_pipe, rf_params)]
scores = []

for name, pipe, params in models:
    grid = GridSearchCV(pipe, params, scoring='roc_auc', cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    print('\n' + name)
    print('Best params:', grid.best_params_)
    print('CV ROC-AUC:', round(grid.best_score_, 4))
    print(confusion_matrix(y_test, y_pred))

    report = pd.DataFrame(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], output_dict=True, zero_division=0)).T
    print(report.loc[['Legit', 'Fraud'], ['precision', 'recall', 'f1-score', 'support']].round(3))

    scores.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

print('\nFinal comparison')
print(pd.DataFrame(scores).round(3).to_string(index=False))
save_path = Path.home() / 'Downloads' / 'cleaned_project2.csv'
df.to_csv(save_path, index=False)
print('Cleaned file saved at:', save_path)