# Predicting Customer Churn in E-Commerce

## Overview
This project develops machine learning models to identify customers of a UK-based online retailer who are at risk of leaving the platform. By recognizing churn early, the retailer can intervene with targeted retention strategies and improve long-term profitability.

## Dataset
The analysis uses the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset, which captures detailed transaction records such as invoice number, stock code, description, quantity, invoice date, unit price, customer ID, and country. Missing `Customer ID` values are removed, product descriptions are imputed with the mode, and transactions with negative quantities (returns or cancellations) are excluded. Feature engineering adds metrics like **TotalPrice** (`Quantity * Price`) and **Days Since Last Purchase** to model purchasing behavior.

## Modeling Approach
Three algorithms are explored to predict churn:
- **Random Forest** – ensemble of decision trees that reduces variance through bagging.
- **Support Vector Machine (SVM)** – finds a hyperplane that maximizes the margin between churners and non-churners; both linear and radial basis function (RBF) kernels are tested.
- **XGBoost** – gradient-boosted decision trees optimized for speed and accuracy.

Data is standardized with `StandardScaler`, split into 70% training and 30% testing subsets, and evaluated with 5-fold cross-validation. Feature importance is examined using model-specific techniques such as intrinsic importance scores for Random Forest, coefficients for linear SVM, and permutation importance for non-linear models.

## Results
Model performance is assessed using accuracy, sensitivity, specificity, and cross-validation scores:
- **XGBoost** achieves accuracy around 66–69% with sensitivity up to ~75% after tuning.
- **Random Forest** improves from ~65% to 69% accuracy with tuning and demonstrates balanced sensitivity and specificity.
- **SVM (Linear)** yields the highest sensitivity (~87%) but lower specificity (~50%), while **SVM (RBF)** offers a better balance and the best average cross-validation score (~70%).

Across models, **PurchaseFrequency** emerges as a key predictor, with `TotalSpent` also prominent in tree-based methods.

## Conclusion
Each algorithm provides distinct strengths: XGBoost and Random Forest offer balanced metrics, whereas SVM maximizes sensitivity. Choice of model depends on whether minimizing false negatives or overall balance is prioritized.

## Repository Structure
- `code/` – Python scripts for data preprocessing, model training, and evaluation.
- `data/` – Dataset files; extract `combined_data.zip` before running the scripts and adjust file paths as needed.
- `figures/` – Images used in the accompanying report.

## Running the Code
1. Unzip `data/combined_data.zip` to obtain `combined_data.csv`.
2. Update the `file_path` variable in `code/Final_project_Terikuti.py` to point to the dataset.
3. Execute:
   ```bash
   python code/Final_project_Terikuti.py
   ```

## Citation
This README summarizes material from the accompanying report `Predicting-Customer-Churn-in-E-Commerce.pdf` by Venkatesh Terikuti.
