# Wine-Quality-Classification-using-decision-tree-and-random-forest
# 🍷 Wine Quality Classification using Decision Tree & Random Forest

Predicting whether a red wine is "good" (quality ≥ 7) or "not good" (quality ≤ 6)** from its physicochemical properties, and comparing a Decision Tree with a Random Forest**.

## Dataset
- Source: [UCI Wine Quality – Red Wine](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009, *Decision Support Systems* 47(4):547–553)
- Size: 1,599 samples → 1,359 after removing 240 duplicate rows
- Features (11): fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- Target: `good = 1` if quality ≥ 7, else `0` (only ~13.5% of wines are "good" → **imbalanced**)

## Workflow
1. Load data, check missing values and duplicates (0 missing, 240 duplicates removed)
2. EDA – quality distribution, correlation heatmap
3. Convert quality score into a binary target
4. Stratified 80/20 train-test split (`random_state=42`)
5. Train baseline Decision Tree and Random Forest (`class_weight="balanced"` to handle imbalance)
6. Hyperparameter tuning with `GridSearchCV` (5-fold stratified CV, scoring = F1)
7. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, 5-fold CV
8. Feature importance analysis

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (baseline) | 0.846 | 0.432 | 0.432 | 0.432 | 0.672 |
| Decision Tree (tuned) | 0.754 | 0.333 | 0.811 | 0.472 | 0.850 |
| Random Forest (baseline) | 0.864 | 0.500 | 0.486 | 0.493 | 0.876 |
| Random Forest (tuned) | 0.864 | 0.500 | 0.622 | 0.554 | 0.882 |

5-fold CV F1: Decision Tree 0.444 vs Random Forest 0.563.

Best params – DT: `max_depth=3, criterion=gini` · RF: `n_estimators=200, max_depth=10, min_samples_leaf=2, max_features=sqrt`

### Key takeaways
- Random Forest beats Decision Tree** on almost every metric, and is more stable in cross-validation.
- The tuned Decision Tree has high recall but low precision (many false "good" predictions) – a shallow tree underfits.
- Accuracy is misleading here (a model that always says "not good" gets ~86%), so F1 and ROC-AUC on the minority class matter more.
- Top features: alcohol (0.247), sulphates (0.148), citric acid (0.106), volatile acidity (0.099), total sulfur dioxide (0.073).

## Report
Full write-up (abstract, methods, results, limitations, references): [`report/Wine_Quality_Classification_Report.pdf`](report/Wine_Quality_Classification_Report.pdf). Executed notebook: [`wine_quality_classification.ipynb`](wine_quality_classification.ipynb).

## Visuals
| | |
|---|---|
| ![](images/quality_distribution.png) | ![](images/correlation_heatmap.png) |
| ![](images/confusion_matrices.png) | ![](images/roc_curves.png) |
| ![](images/feature_importance.png) | ![](images/decision_tree.png) |

 Run it yourself
```bash
git clone https://github.com/anidata3099/wine-quality-classification.git
cd wine-quality-classification
pip install -r requirements.txt
python wine_quality_classification.py
```

## Possible improvements
- Try SMOTE / threshold tuning for the minority class
- Compare with Gradient Boosting / XGBoost
- Include the white wine dataset

 Tech stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Team 
Anirban Roy (111223099)
Israt Jahan sumaiya (111223086)

