
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance


# Load the dataset
file_path = 'C:/Users/ual-laptop/ML project/combined_data.csv' 
data = pd.read_csv(file_path)

# Handling Missing Values

# Check for missing values in each column
missing_values = data.isnull().sum()
print(missing_values)

# Remove rows where 'Customer ID' is missing
data = data.dropna(subset=['Customer ID'])

# Replace missing values in 'Description' with the mode
mode_description = data['Description'].mode()[0]
data['Description'] = data['Description'].fillna(mode_description)

missing_values = data.isnull().sum()
print(missing_values)

# Exclude rows where 'Quantity' is negative
data = data[data['Quantity'] > 0]

# Data Preprocessing
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['TotalPrice'] = data['Quantity'] * data['Price']
customer_data = data.groupby('Customer ID').agg(
    TotalSpent=pd.NamedAgg(column='TotalPrice', aggfunc='sum'),
    AverageTransactionValue=pd.NamedAgg(column='TotalPrice', aggfunc='mean'),
    PurchaseFrequency=pd.NamedAgg(column='Invoice', aggfunc=pd.Series.nunique),
    LastPurchaseDate=pd.NamedAgg(column='InvoiceDate', aggfunc='max')
).reset_index()  # Resetting the index here

latest_date = data['InvoiceDate'].max()
customer_data['DaysSinceLastPurchase'] = (latest_date - customer_data['LastPurchaseDate']).dt.days

# Churn label creation (assuming churn if no purchase in the last 90 days)
churn_threshold = 90
customer_data['Churn'] = customer_data['DaysSinceLastPurchase'] > churn_threshold

# Data for modeling
X = customer_data.drop(['Customer ID', 'LastPurchaseDate', 'Churn', 'DaysSinceLastPurchase'], axis=1)
y = customer_data['Churn']
missing_values = X.isnull().sum()
print(missing_values)
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)

# Evaluation
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, xgb_predictions).ravel()
xgb_sensitivity = tp / (tp + fn)
xgb_specificity = tn / (tn + fp)

# Printing the evaluation metrics
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Sensitivity:", xgb_sensitivity)
print("XGBoost Specificity:", xgb_specificity)

# Random Forest Model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train_scaled, y_train)
rf_predictions = random_forest_model.predict(X_test_scaled)

# Calculate metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, rf_predictions).ravel()
rf_sensitivity = tp / (tp + fn)
rf_specificity = tn / (tn + fp)

# Print the results
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Sensitivity:", rf_sensitivity)
print("Random Forest Specificity:", rf_specificity)

# SVM Model with Linear Kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

# SVM Model with Gaussian Kernel
svm_model_g = SVC(kernel='rbf')
svm_model_g.fit(X_train_scaled, y_train)
svm_predictions_g = svm_model_g.predict(X_test_scaled)

# Calculate metrics for SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, svm_predictions).ravel()
svm_sensitivity = tp / (tp + fn)
svm_specificity = tn / (tn + fp)

# Calculate metrics for SVM Gaussian
svm_accuracy_g = accuracy_score(y_test, svm_predictions_g)
tn, fp, fn, tp = confusion_matrix(y_test, svm_predictions_g).ravel()
svm_sensitivity_g = tp / (tp + fn)
svm_specificity_g = tn / (tn + fp)

# Print the results
print("SVM Accuracy:", svm_accuracy)
print("SVM Sensitivity:", svm_sensitivity)
print("SVM Specificity:", svm_specificity)
print("SVM Accuracy_g:", svm_accuracy_g)
print("SVM Sensitivity_g:", svm_sensitivity_g)
print("SVM Specificity_g:", svm_specificity_g)


# Models
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier()
svm_model = SVC(kernel='linear')

# Cross-validation (5 folds)
cv_folds = 5

# XGBoost Cross-Validation
xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv_folds)
print("XGBoost Cross-Validation Scores:", xgb_cv_scores)
print("XGBoost Average CV Score:", xgb_cv_scores.mean())

# Random Forest Cross-Validation
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv_folds)
print("Random Forest Cross-Validation Scores:", rf_cv_scores)
print("Random Forest Average CV Score:", rf_cv_scores.mean())

# SVM Cross-Validation
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=cv_folds)
print("SVM Cross-Validation Scores:", svm_cv_scores)
print("SVM Average CV Score:", svm_cv_scores.mean())

# XGBoost Model
# Fine-tuning hyperparameters using GridSearchCV

xgb_params = {
    'objective': ['binary:logistic'],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5, 8],
    'n_estimators': [100, 200]
}

xgb_model = xgb.XGBClassifier()
grid_search_xgb = GridSearchCV(xgb_model, param_grid=xgb_params, scoring='accuracy', cv=5)
grid_search_xgb.fit(X_train_scaled, y_train)

# Get the best model from the grid search
xgb_model = grid_search_xgb.best_estimator_

xgb_predictions = xgb_model.predict(X_test_scaled)

# Evaluation
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, xgb_predictions).ravel()
xgb_sensitivity = tp / (tp + fn)
xgb_specificity = tn / (tn + fp)

# Print the evaluation metrics
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Sensitivity:", xgb_sensitivity)
print("XGBoost Specificity:", xgb_specificity)

# Define cross-validation parameters
cv_folds = 5  # Specify the number of folds

# XGBoost Model (with optimized hyperparameters)
xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv_folds)
print("XGBoost Cross-Validation Scores:", xgb_cv_scores)
print("XGBoost Average CV Score:", xgb_cv_scores.mean())

# Random Forest Model
# Fine-tuning hyperparameters using GridSearchCV

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 8],
    'min_samples_leaf': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf_model, param_grid=rf_params, scoring='accuracy', cv=5)
grid_search_rf.fit(X_train_scaled, y_train)

# Get the best model from the grid search
rf_model = grid_search_rf.best_estimator_

rf_predictions = rf_model.predict(X_test_scaled)

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, rf_predictions).ravel()
rf_sensitivity = tp / (tp + fn)
rf_specificity = tn / (tn + fp)

# Print the evaluation metrics
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Sensitivity:", rf_sensitivity)
print("Random Forest Specificity:", rf_specificity)

# Random Forest Model (with optimized hyperparameters)
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv_folds)
print("Random Forest Cross-Validation Scores:", rf_cv_scores)
print("Random Forest Average CV Score:", rf_cv_scores.mean())

# SVM Model with Linear Kernel
# Fine-tuning hyperparameters using GridSearchCV

svm_params = {
    'kernel': ['linear'],
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.001]
}

svm_model = SVC()
grid_search_svm = GridSearchCV(svm_model, param_grid=svm_params, scoring='accuracy', cv=5)
grid_search_svm.fit(X_train_scaled, y_train)

# Get the best model from the grid search
svm_model = grid_search_svm.best_estimator_

svm_predictions = svm_model.predict(X_test_scaled)

# Evaluation
svm_accuracy = accuracy_score(y_test, svm_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, svm_predictions).ravel()
svm_sensitivity = tp / (tp + fn)
svm_specificity = tn / (tn + fp)

# Print the evaluation metrics
print("SVM Accuracy:", svm_accuracy)
print("SVM Sensitivity:", svm_sensitivity)
print("SVM Specificity:", svm_specificity)

# SVM Model with Linear Kernel (with optimized hyperparameters)
svm_linear_cv_scores = cross_val_score(grid_search_svm.best_estimator_, X_train_scaled, y_train, cv=cv_folds)
print("SVM with Linear Kernel Cross-Validation Scores:", svm_linear_cv_scores)
print("SVM with Linear Kernel Average CV Score:", svm_linear_cv_scores.mean())

# SVM Model with Gaussian Kernel
# Fine-tuning hyperparameters using GridSearchCV

svm_params = {
    'kernel': ['rbf'], # Specify Gaussian kernel as 'rbf'
    'C': [0.1, 1, 10], # Regularization parameter
    'gamma': [0.01, 0.001, 0.1] # Kernel coefficient
}

svm_model = SVC()
grid_search_svm_gaussian = GridSearchCV(svm_model, param_grid=svm_params, scoring='accuracy', cv=5)
grid_search_svm_gaussian.fit(X_train_scaled, y_train)

# Get the best model from the grid search
svm_model = grid_search_svm_gaussian.best_estimator_

# Make predictions
svm_predictions = svm_model.predict(X_test_scaled)

# Evaluate the model
svm_accuracy = accuracy_score(y_test, svm_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, svm_predictions).ravel()
svm_sensitivity = tp / (tp + fn)
svm_specificity = tn / (tn + fp)

# Print the evaluation metrics
print("SVM with Gaussian Kernel Accuracy:", svm_accuracy)
print("SVM with Gaussian Kernel Sensitivity:", svm_sensitivity)
print("SVM with Gaussian Kernel Specificity:", svm_specificity)

# SVM Model with Gaussian Kernel (with optimized hyperparameters)
svm_gaussian_cv_scores = cross_val_score(grid_search_svm_gaussian.best_estimator_, X_train_scaled, y_train, cv=cv_folds)
print("SVM with Gaussian Kernel Cross-Validation Scores:", svm_gaussian_cv_scores)
print("SVM with Gaussian Kernel Average CV Score:", svm_gaussian_cv_scores.mean())

# Training the XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# Feature Importance
feature_importances = xgb_model.feature_importances_

# Creating a bar plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(feature_importances)), [X.columns[i] for i in range(len(feature_importances))])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Analysis with XGBoost')
plt.show()

# Random Forest feature importances
rf_feature_importances = random_forest_model.feature_importances_

# Plotting feature importances for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(range(len(rf_feature_importances)), rf_feature_importances)
plt.yticks(range(len(rf_feature_importances)), X.columns)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Analysis with Random Forest')
plt.show()


# Plotting feature importances for Linear SVM model
if hasattr(svm_model, "coef_"):
    print("Plotting feature importances for Linear SVM...")
    svm_linear_coefficients = np.abs(svm_model.coef_[0])

    # Plotting feature importances for Linear SVM
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(svm_linear_coefficients)), svm_linear_coefficients)
    plt.yticks(range(len(svm_linear_coefficients)), X.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Analysis with Linear SVM')
    plt.show()

# Plotting feature importances for non Linear SVM model
# Calculate permutation importance
perm_importance = permutation_importance(svm_model_g, X_test_scaled, y_test, n_repeats=30, random_state=42)

# Sort the importances in descending order
sorted_idx = perm_importance.importances_mean.argsort()

# Plotting the permutation importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Permutation Importance with Non-Linear SVM (RBF Kernel)')
plt.show()
