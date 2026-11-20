"""
Wine Quality Classification using Decision Tree and Random Forest
Dataset: UCI Red Wine Quality (1599 samples, 11 physicochemical features)
Target : binary -> good (quality >= 7) vs not good (quality <= 6)
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

RS = 42
df = pd.read_csv("data/winequality-red.csv", sep=";")
print("Shape:", df.shape, "| Missing:", int(df.isna().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))
df = df.drop_duplicates().reset_index(drop=True)

# ---------- EDA ----------
plt.figure(figsize=(6, 4))
sns.countplot(x="quality", data=df, color="#7b2d43")
plt.title("Quality score distribution"); plt.tight_layout()
plt.savefig("images/quality_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=150); plt.close()

# ---------- Target ----------
df["good"] = (df["quality"] >= 7).astype(int)
print("\nClass balance:\n", df["good"].value_counts(normalize=True).round(3))
X, y = df.drop(columns=["quality", "good"]), df["good"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

def evaluate(name, model):
    p = model.predict(X_te); pr = model.predict_proba(X_te)[:, 1]
    r = dict(Model=name, Accuracy=accuracy_score(y_te, p), Precision=precision_score(y_te, p),
             Recall=recall_score(y_te, p), F1=f1_score(y_te, p), ROC_AUC=roc_auc_score(y_te, pr))
    print(f"\n=== {name} ===\n", classification_report(y_te, p, target_names=["Not good", "Good"]))
    return r, p, pr

results, preds = [], {}

# ---------- Decision Tree ----------
dt_base = DecisionTreeClassifier(class_weight="balanced", random_state=RS).fit(X_tr, y_tr)
r, p, pr = evaluate("Decision Tree (baseline)", dt_base); results.append(r)

dt_grid = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=RS),
    {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [1, 5, 10, 20], "criterion": ["gini", "entropy"]},
    scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
dt = dt_grid.best_estimator_
print("Best DT params:", dt_grid.best_params_)
r, p, pr = evaluate("Decision Tree (tuned)", dt); results.append(r); preds["Decision Tree"] = (p, pr)

# ---------- Random Forest ----------
rf_base = RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=-1).fit(X_tr, y_tr)
r, p, pr = evaluate("Random Forest (baseline)", rf_base); results.append(r)

rf_grid = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=-1),
    {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4],
     "max_features": ["sqrt", "log2"]}, scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)
r, p, pr = evaluate("Random Forest (tuned)", rf); results.append(r); preds["Random Forest"] = (p, pr)

res = pd.DataFrame(results).round(3)
res.to_csv("results.csv", index=False)
print("\n", res.to_string(index=False))
print("\n5-fold CV F1 -> DT: %.3f | RF: %.3f" % (cross_val_score(dt, X, y, cv=cv, scoring="f1").mean(),
                                               cross_val_score(rf, X, y, cv=cv, scoring="f1").mean()))

# ---------- Plots ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (n, (p, _)) in zip(ax, preds.items()):
    sns.heatmap(confusion_matrix(y_te, p), annot=True, fmt="d", cmap="Reds", ax=a,
                xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"])
    a.set_title(n); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("images/confusion_matrices.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
for n, (_, pr) in preds.items():
    fpr, tpr, _ = roc_curve(y_te, pr); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te, pr):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves"); plt.legend()
plt.tight_layout(); plt.savefig("images/roc_curves.png", dpi=150); plt.close()

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5)); imp.plot.barh(color="#7b2d43"); plt.title("Random Forest feature importance")
plt.tight_layout(); plt.savefig("images/feature_importance.png", dpi=150); plt.close()
print("\nTop features:\n", imp.sort_values(ascending=False).head(5).round(3))

plt.figure(figsize=(18, 8))
plot_tree(dt, feature_names=X.columns, class_names=["Not good", "Good"], filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (top 3 levels)"); plt.tight_layout()
plt.savefig("images/decision_tree.png", dpi=130); plt.close()

# ---------- Save numbers used in the report ----------
import json
cm = {n: confusion_matrix(y_te, p).tolist() for n, (p, _) in preds.items()}
json.dump({"n_train": len(X_tr), "n_test": len(X_te), "n_after_dedup": len(df),
           "pos_rate": float(y.mean()), "confusion": cm,
           "dt_params": dt_grid.best_params_, "rf_params": rf_grid.best_params_,
           "cv_dt": float(cross_val_score(dt, X, y, cv=cv, scoring="f1").mean()),
           "cv_rf": float(cross_val_score(rf, X, y, cv=cv, scoring="f1").mean()),
           "importance": imp.sort_values(ascending=False).round(4).to_dict(),
           "quality_counts": df["quality"].value_counts().sort_index().to_dict()},
          open("metrics.json", "w"), indent=2)
df.drop(columns=["good"]).describe().T[["mean","std","min","max"]].round(3).to_csv("feature_summary.csv")
