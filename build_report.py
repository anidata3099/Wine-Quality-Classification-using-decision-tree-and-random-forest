import json, csv, os
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether)

AUTHOR = "Pollob"
DATE = "20 November 2026"
REPO = "https://github.com/anidata3099/wine-quality-classification"
UCI = "https://archive.ics.uci.edu/dataset/186/wine+quality"

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("DS", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DS-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

M = json.load(open("metrics.json"))
BURG = colors.HexColor("#7b2d43")
body = ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="DS-B", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, textColor=BURG, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="DS-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4, keepWithNext=1)
title = ParagraphStyle("title", fontName="DS-B", fontSize=18, leading=23, alignment=TA_CENTER, textColor=BURG, spaceAfter=6)
sub = ParagraphStyle("sub", fontName="DS", fontSize=10, leading=14, alignment=TA_CENTER, spaceAfter=3)
cap = ParagraphStyle("cap", fontName="DV-I", fontSize=8.6, leading=12, alignment=TA_JUSTIFY, spaceAfter=10, spaceBefore=3)
tcap = ParagraphStyle("tcap", parent=cap, spaceBefore=8, spaceAfter=3)
cell = ParagraphStyle("cell", fontName="DS", fontSize=8, leading=10)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DS-B")
ref = ParagraphStyle("ref", fontName="DV", fontSize=8.8, leading=12.5, leftIndent=14, firstLineIndent=-14, spaceAfter=4)
absst = ParagraphStyle("abs", parent=body, leftIndent=18, rightIndent=18, fontSize=9.2, leading=13.2)

def P(t, s=body): return Paragraph(t, s)

def fig(path, width_cm, caption):
    w, h = PILImage.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, cap)])

def table(data, widths, caption, bold_row=None):
    rows = [[Paragraph(str(c), cellb if (i == 0 or i == bold_row) else cell) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0dfe4")),
          ("LINEABOVE", (0, 0), (-1, 0), 0.8, BURG), ("LINEBELOW", (0, 0), (-1, 0), 0.5, BURG),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, BURG), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if bold_row: st.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#faf3f5")))
    t.setStyle(TableStyle(st))
    return KeepTogether([P(caption, tcap), t, Spacer(1, 8)])

def footer(c, d):
    c.saveState(); c.setFont("DS", 8); c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 1.2 * cm, f"Wine Quality Classification - Decision Tree vs Random Forest  |  Page {d.page}")
    c.restoreState()

cm_dt, cm_rf = M["confusion"]["Decision Tree"], M["confusion"]["Random Forest"]
f = lambda x: f"{x:.3f}"
S = []

# ---------------- Title + abstract ----------------
S += [Spacer(1, 6), P("Wine Quality Classification using Decision Tree and Random Forest", title),
      P(f"{AUTHOR}", sub), P(f"Department of Computer Science and Engineering &nbsp;|&nbsp; {DATE}", sub),
      P(f'Code and notebook: <link href="{REPO}" color="#7b2d43">{REPO}</link>', sub), Spacer(1, 10)]
S.append(P("<b>Abstract</b>", h2))
S.append(P(
    "This project predicts whether a red wine is of high quality (sensory score &#8805; 7) from 11 physicochemical "
    "measurements, using the UCI Wine Quality dataset (Cortez et al., 2009). After removing 240 duplicate rows, 1,359 "
    "samples remained, of which only 13.5% were high quality. A CART Decision Tree (Breiman et al., 1984) and a Random "
    "Forest (Breiman, 2001) were trained with class weighting and tuned by 5-fold stratified cross-validation on F1. "
    f"On a held-out stratified test set of {M['n_test']} wines, the tuned Random Forest reached accuracy 0.864, F1 0.554 and "
    f"ROC-AUC 0.882, compared with accuracy 0.754, F1 0.472 and ROC-AUC 0.850 for the tuned Decision Tree. Cross-validated F1 "
    f"confirmed the ranking ({M['cv_rf']:.3f} vs {M['cv_dt']:.3f}). Alcohol, sulphates, citric acid and volatile acidity were the most "
    "influential features. Because the minority class is small, absolute F1 remains modest and the comparison should be read "
    "with the uncertainty discussed in Section 7.", absst))
S.append(P("<b>Keywords:</b> wine quality, classification, decision tree, random forest, class imbalance, feature importance", absst))

# ---------------- 1 Intro ----------------
S.append(P("1. Introduction", h1))
S.append(P(
    "Wine quality is normally judged by human tasters, which is slow, costly and subjective. If quality can be predicted "
    "from routine laboratory measurements, producers could screen batches earlier. This project treats the problem as "
    "<b>binary classification</b> (good vs. not good) and compares two tree-based learners: a single, interpretable "
    "Decision Tree and a Random Forest, which averages many decorrelated trees to reduce variance. The questions are "
    "(i) how much better the ensemble is than the single tree, and (ii) which chemical properties drive the predictions."))

# ---------------- 2 Dataset ----------------
S.append(P("2. Dataset", h1))
S.append(P(
    f'The data are the red-wine subset of the Wine Quality dataset (Cortez et al., 2009), distributed through the UCI Machine '
    f'Learning Repository (<link href="{UCI}" color="#7b2d43">{UCI}</link>). It contains 1,599 Portuguese "Vinho Verde" red wines, '
    "each described by 11 numeric physicochemical features and a sensory quality score from 0 to 10 (median of at least three "
    "tasters). Observed scores range from 3 to 8. There are no missing values. Table 1 summarises the features."))
rows = [["Feature", "Mean", "Std", "Min", "Max"]]
for r in csv.DictReader(open("feature_summary.csv")):
    rows.append([r[""], r["mean"], r["std"], r["min"], r["max"]])
S.append(table(rows, [5, 2.4, 2.4, 2.4, 2.4], "Table 1. Descriptive statistics of the 11 input features and the quality score (n = 1,359 after duplicate removal)."))
S.append(P(
    "Table 1 shows that features live on very different scales (total sulfur dioxide up to 289, density about 1.0) and that "
    "several are right-skewed with extreme values (chlorides, residual sugar, sulphates). Tree-based models split on thresholds "
    "and are invariant to monotonic rescaling, so no scaling was applied and outliers were kept."))

# ---------------- 3 Preprocessing ----------------
S.append(P("3. Preprocessing and Experimental Setup", h1))
S.append(P("All steps are implemented in <font name='DV-I'>wine_quality_classification.py</font> and the accompanying notebook, with <font name='DV-I'>random_state = 42</font> everywhere, so results are reproducible."))
steps = [
    "<b>Missing values and duplicates.</b> No missing values were found. The file contains 240 exact duplicate rows; they were removed (1,599 &#8594; 1,359) <i>before</i> splitting, because duplicates falling on both sides of a split would leak test information into training.",
    "<b>Target construction.</b> The score is converted to a binary label: <i>good = 1</i> if quality &#8805; 7, otherwise 0. This gives 184 good wines (13.5%) and 1,175 not good (86.5%), see Figure 1.",
    f"<b>Split.</b> A stratified 80/20 train-test split gives {M['n_train']} training and {M['n_test']} test samples (37 good wines in the test set). The test set is used only once, for final evaluation.",
    "<b>Imbalance handling.</b> Both models use <i>class_weight = 'balanced'</i>, which weights errors on the minority class inversely to its frequency. No resampling was used.",
    "<b>Hyperparameter tuning.</b> GridSearchCV with 5-fold stratified cross-validation on the training set only, optimising F1 of the good class (Table 2).",
]
for i, s in enumerate(steps, 1): S.append(P(f"{i}. {s}"))
S.append(table([["Model", "Grid searched", "Selected"],
    ["Decision Tree", "max_depth {3,4,5,6,8,10}; min_samples_leaf {1,5,10,20}; criterion {gini, entropy}", "max_depth = 3, min_samples_leaf = 1, criterion = gini"],
    ["Random Forest", "n_estimators {200,400}; max_depth {None,10,20}; min_samples_leaf {1,2,4}; max_features {sqrt, log2}", "n_estimators = 200, max_depth = 10, min_samples_leaf = 2, max_features = sqrt"]],
    [2.6, 7.4, 6.6], "Table 2. Hyperparameter search space and the configuration selected by cross-validated F1."))

# ---------------- 4 Methods ----------------
S.append(P("4. Models and Evaluation Metrics", h1))
S.append(P(
    "<b>Decision Tree.</b> scikit-learn's <i>DecisionTreeClassifier</i> (Pedregosa et al., 2011) implements an optimised CART algorithm "
    "(Breiman et al., 1984): it recursively chooses the feature and threshold that most reduce node impurity (Gini or entropy) and stops at "
    "the configured depth or leaf size. It is easy to read but has high variance."))
S.append(P(
    "<b>Random Forest.</b> <i>RandomForestClassifier</i> builds many trees, each on a bootstrap sample and with a random subset of features "
    "considered at each split, and averages their predicted class probabilities (Breiman, 2001). This decorrelates the trees and reduces variance. "
    "Feature importance is the mean decrease in impurity attributable to each feature."))
S.append(P(
    "<b>Metrics.</b> Since only 13.5% of wines are good, accuracy alone is misleading: always predicting &#8220;not good&#8221; scores 86.5%. "
    "We therefore report precision, recall and F1 for the good class, ROC-AUC (threshold-independent ranking quality), the confusion matrix, "
    "and 5-fold cross-validated F1 on the full data."))

# ---------------- 5 Results ----------------
S.append(P("5. Results", h1))
S.append(P("5.1 Exploratory analysis", h2))
S.append(fig("images/quality_distribution.png", 8.2, "Figure 1. Distribution of wine quality scores. Most wines score 5 or 6; only scores 7 and 8 (the &#8220;good&#8221; class) are rare, and there are almost no wines at the extremes."))
S.append(P("Figure 1 confirms the class imbalance that motivates class weighting and F1-based evaluation. The score is concentrated at 5 and 6, so the boundary between &#8220;good&#8221; (&#8805; 7) and &#8220;not good&#8221; sits in a sparsely populated tail."))
S.append(fig("images/correlation_heatmap.png", 11.5, "Figure 2. Pearson correlation matrix of all features and quality. Alcohol has the strongest positive correlation with quality, volatile acidity the strongest negative."))
S.append(P(
    "In Figure 2, alcohol (r = 0.48), sulphates (0.25) and citric acid (0.23) correlate positively with quality, while volatile acidity "
    "(&#8722;0.40), density (&#8722;0.18) and total sulfur dioxide (&#8722;0.18) correlate negatively. Several features are also strongly inter-correlated: "
    "fixed acidity with pH (&#8722;0.69) and with citric acid (0.67), free with total sulfur dioxide (0.67), and alcohol with density (&#8722;0.51). "
    "Such collinearity does not harm prediction in tree models, but it splits importance among related features, which matters when interpreting Figure 5."))

S.append(P("5.2 Model performance", h2))
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
vals = [("Decision Tree (baseline)", .846, .432, .432, .432, .672), ("Decision Tree (tuned)", .754, .333, .811, .472, .850),
        ("Random Forest (baseline)", .864, .500, .486, .493, .876), ("Random Forest (tuned)", .864, .500, .622, .554, .882)]
for v in vals: rows.append([v[0]] + [f(x) for x in v[1:]])
S.append(table(rows, [5, 2.2, 2.2, 2.2, 2.2, 2.2], f"Table 3. Test-set performance (n = {M['n_test']}; metrics for the good class). Baselines use default hyperparameters with balanced class weights; tuned models use the settings in Table 2.", bold_row=4))
S.append(P(
    f"Table 3 shows that the tuned Random Forest is best on every threshold-dependent metric except recall, and has the highest ROC-AUC (0.882). "
    f"Five-fold cross-validated F1 on all data agrees: {M['cv_rf']:.3f} for the Random Forest vs. {M['cv_dt']:.3f} for the tuned Decision Tree, so the advantage is not an artefact of one split."))
S.append(fig("images/confusion_matrices.png", 13, f"Figure 3. Confusion matrices of the tuned models on the test set. Decision Tree: TN = {cm_dt[0][0]}, FP = {cm_dt[0][1]}, FN = {cm_dt[1][0]}, TP = {cm_dt[1][1]}. Random Forest: TN = {cm_rf[0][0]}, FP = {cm_rf[0][1]}, FN = {cm_rf[1][0]}, TP = {cm_rf[1][1]}."))
S.append(P(
    f"Figure 3 shows <i>how</i> the models differ. The shallow Decision Tree finds {cm_dt[1][1]} of the 37 good wines (recall 0.81) but mislabels {cm_dt[0][1]} ordinary wines as good, giving low precision (0.33) and low accuracy (0.754). "
    f"The Random Forest is more conservative: it finds {cm_rf[1][1]} good wines (recall 0.62) with only {cm_rf[0][1]} false positives (precision 0.50). Which model is preferable depends on cost: "
    "a producer who wants to <i>not miss</i> any premium batch might accept the tree's false alarms, whereas one who must verify every flagged batch by hand would prefer the forest."))
S.append(fig("images/roc_curves.png", 8.2, "Figure 4. ROC curves of the tuned models on the test set. The dashed line is a random classifier."))
S.append(P(
    "In Figure 4 the Random Forest curve lies above the tree's over most of the range (AUC 0.882 vs. 0.850), meaning it ranks good wines above ordinary ones more reliably across all thresholds. "
    "The tree's curve is a coarse staircase because a depth-3 tree produces only a handful of distinct probability values. Notably, tuning raised the tree's AUC from 0.672 to 0.850 (Table 3): "
    "the default fully grown tree overfits and its probabilities are nearly all 0 or 1."))

S.append(P("5.3 Feature importance", h2))
S.append(fig("images/feature_importance.png", 9.5, "Figure 5. Random Forest feature importance (mean decrease in impurity). Alcohol is the most influential feature, followed by sulphates, citric acid and volatile acidity."))
imp = M["importance"]
S.append(P(
    f"Figure 5 ranks alcohol first (importance {imp['alcohol']:.3f}), then sulphates ({imp['sulphates']:.3f}), citric acid ({imp['citric acid']:.3f}), volatile acidity ({imp['volatile acidity']:.3f}) and total sulfur dioxide ({imp['total sulfur dioxide']:.3f}). "
    "This agrees with the correlation analysis in Figure 2 and with the wine-chemistry literature: higher alcohol is associated with fuller body and higher ratings, while high volatile acidity (an acetic, vinegar-like note) is a known fault. "
    "Impurity-based importance is biased toward continuous and high-cardinality features and is shared among correlated features, so it indicates association with the label, not causation."))
S.append(fig("images/decision_tree.png", 16.2, "Figure 6. Top three levels of the tuned Decision Tree (max_depth = 3). Each node shows the split rule, Gini impurity, sample count, class counts and majority class."))
S.append(P(
    "The tree in Figure 6 is directly readable. Its root split is on alcohol (&#8804; 10.45), the same feature ranked first by the forest. "
    "Wines with alcohol above 10.45 and sulphates above 0.62 are all predicted good, while low-alcohol wines are predicted good only when fixed acidity is above 11.55 and sulphates above 0.66. "
    "Because this single rule covers a large share of the higher-alcohol wines, it explains the tree's many false positives in Figure 3 (60 ordinary wines flagged as good): alcohol and sulphates raise the odds of a good rating but do not guarantee it. "
    "Volatile acidity, which ranks fourth in Figure 5, does not appear in the top three levels, so the tree uses less of the available signal than the forest. "
    "The interpretability of the tree is its main advantage and is lost in the 200-tree forest."))

# ---------------- 6 Discussion ----------------
S.append(P("6. Discussion", h1))
S.append(P(
    "The ensemble outperforms the single tree on F1 (0.554 vs. 0.472), ROC-AUC (0.882 vs. 0.850) and cross-validated F1 (0.563 vs. 0.444), which is consistent with the variance-reduction argument for bagging. "
    "However, an F1 of about 0.55 means the forest still misses roughly 38% of good wines and about half of the wines it flags are not truly good. The main reasons are (i) the strong imbalance, "
    "(ii) the subjectivity of tasting scores, so that a wine scored 6 and one scored 7 can be chemically almost identical, and (iii) the limited information in 11 chemical measurements, which cannot capture aroma or terroir."))
S.append(P(
    "The results also show why accuracy should not be used alone: the tuned tree has the lowest accuracy yet the highest recall, and a trivial majority-class predictor would reach 86.5% accuracy, slightly above our forest's 86.4%. "
    "The forest's value lies in detecting the minority class, not in raw accuracy."))

# ---------------- 7 Limitations ----------------
S.append(P("7. Limitations", h1))
lims = [
    "<b>Small minority class in the test set.</b> The test set has only 37 good wines, so one extra correct prediction changes recall by 2.7 points. The reported differences, especially between baseline and tuned variants, are within plausible sampling noise. We did not run significance tests or repeat the split with different seeds; only the 5-fold CV partially mitigates this.",
    "<b>Single dataset, single region.</b> Only red Vinho Verde wine from Portugal is used. Results may not transfer to white wine, other regions or other tasting panels.",
    "<b>Subjective, arbitrary label.</b> The &#8805; 7 threshold was chosen by us; other thresholds would change the class balance and the conclusions. The score itself is a noisy human judgement.",
    "<b>Duplicate removal is a judgement call.</b> Some of the 240 duplicates may be different wines with identical measurements rated by different tasters. We removed them to avoid leakage, which also reduces the data available for learning.",
    "<b>Limited tuning and model scope.</b> Only small grids were searched; no threshold tuning, resampling (e.g. SMOTE) or probability calibration was tried, and boosted models were not compared. The tuned tree's depth of 3 may underfit.",
    "<b>Interpretation of importances.</b> Impurity-based importance is biased and shared among correlated features (Figure 2); permutation importance or SHAP values would give a more robust picture.",
]
for l in lims: S.append(P("&#8226; " + l, ParagraphStyle("li", parent=body, leftIndent=12, firstLineIndent=-8, spaceAfter=3)))

# ---------------- 8 Conclusion ----------------
S.append(P("8. Conclusion and Future Work", h1))
S.append(P(
    "A tuned Random Forest classified good red wines better than a tuned Decision Tree (F1 0.554 vs. 0.472, ROC-AUC 0.882 vs. 0.850), and alcohol, sulphates, citric acid and volatile acidity were the most informative features. "
    "Future work includes decision-threshold tuning or SMOTE for the minority class, gradient-boosted models, repeated cross-validation with significance testing, and permutation-based explanations."))

# ---------------- Reproducibility ----------------
S.append(P("Reproducibility", h1))
S.append(P(
    f'Source code, executed Jupyter notebook, dataset file and this report: <link href="{REPO}" color="#7b2d43">{REPO}</link>. '
    f'Original dataset: <link href="{UCI}" color="#7b2d43">{UCI}</link>. Run <font name="DV-I">pip install -r requirements.txt</font> then '
    '<font name="DV-I">python wine_quality_classification.py</font>. Environment: Python 3, scikit-learn 1.9.1, random_state = 42. '
    "The CSV bundled in the repository was obtained from a public GitHub mirror of the UCI file; it has 1,599 rows and the same 12 columns as the original."))

# ---------------- References ----------------
S.append(P("References", h1))
refs = [
    'Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5-32. https://doi.org/10.1023/A:1010933404324',
    'Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.',
    f'Cortez, P., Cerdeira, A., Almeida, F., Matos, T., &amp; Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. <i>Decision Support Systems, 47</i>(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016',
    f'Cortez, P., Cerdeira, A., Almeida, F., Matos, T., &amp; Reis, J. (2009). <i>Wine Quality</i> [Dataset]. UCI Machine Learning Repository. {UCI}',
    'Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825-2830.',
]
for r in refs: S.append(P(r, ref))

doc = SimpleDocTemplate("report/Wine_Quality_Classification_Report.pdf", pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                        topMargin=2*cm, bottomMargin=2*cm, title="Wine Quality Classification using Decision Tree and Random Forest", author=AUTHOR)
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("built")
