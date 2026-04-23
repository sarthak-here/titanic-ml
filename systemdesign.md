# Titanic ML — System Design

## What It Does
A Kaggle competition project solving the Titanic survival prediction problem using five progressively sophisticated machine learning approaches, each in its own module. Demonstrates iterative model improvement from baseline to advanced ensemble techniques.

---

## Architecture

```
titanic_model.py  (orchestrator)
        |
        +------+------+------+------+------+
        |      |      |      |      |      |
       v1     v2     v3     v4     v5
  simple_rf  segment  feature  companion  smote
              ed      _eng     _survival
```

---

## Dataset

```
train.csv  (891 rows) -> used for training + cross-validation
test.csv   (418 rows) -> submission predictions

Features:
  Pclass    (1/2/3 ticket class)
  Sex       (male/female)
  Age       (float, ~20% missing)
  SibSp     (siblings/spouses aboard)
  Parch     (parents/children aboard)
  Fare      (ticket price)
  Embarked  (C/Q/S, 2 missing)
  Cabin     (very sparse, ~77% missing)

Target: Survived (0/1)
```

---

## The Five Models

### v1 — Simple Random Forest (models/v1_simple_rf.py)
```
  Raw features -> median imputation -> LabelEncoding
  RandomForestClassifier(n_estimators=100)
  Baseline CV accuracy: ~80%
  Purpose: establish a working baseline quickly
```

### v2 — Segmented Models (models/v2_segmented.py)
```
  Insight: survival patterns differ by sex and class
  Train separate models per passenger segment:
    - Women (high survival, different feature weights)
    - Men in 1st class
    - Men in 2nd/3rd class
  Combine predictions by segment membership
  Improvement: ~1-2% accuracy gain over v1
```

### v3 — Feature Engineering (models/v3_feature_engineering.py)
```
  New derived features:
    Title       = extracted from Name (Mr, Mrs, Miss, Master, Rare)
    FamilySize  = SibSp + Parch + 1
    IsAlone     = 1 if FamilySize == 1
    Deck        = first letter of Cabin (A/B/C/.../Unknown)
    FareBand    = pd.qcut(Fare, 4) -> ordinal
    AgeBand     = pd.cut(Age, 5) -> ordinal
  GradientBoostingClassifier
  Improvement: ~2-3% gain, especially Age handling
```

### v4 — Companion Survival (models/v4_companion_survival.py)
```
  Insight: families and groups had correlated survival
  Feature: look up if ticket-mates survived (train set)
  Ticket grouping -> group survival rate as feature
  XGBoost + companion feature
  Note: requires careful train/test separation to avoid leakage
```

### v5 — SMOTE (models/v5_smote.py)
```
  Problem: training set is slightly imbalanced (62% perished)
  SMOTE oversampling on minority class (survived=1)
  After balancing: train on balanced set
  RandomForest + GridSearchCV tuning
  Focus: improve recall on survivors (reduces false negatives)
```

---

## Data Flow (per model)

```
train.csv + test.csv
        |
  pandas read_csv
        |
  Preprocessing:
  - Age: median imputation (or group median by Title)
  - Embarked: mode imputation (S)
  - Fare: median imputation (test set)
  - Cabin: extract Deck or flag as Unknown
        |
  Feature engineering (model-specific)
        |
  Train/val split (80/20) or StratifiedKFold(5)
        |
  Fit model
        |
  CV accuracy + confusion matrix + feature importance
        |
  Predict on test.csv
        |
  submission.csv  (PassengerId, Survived)
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| One file per model version | Easy to compare approaches; no shared state between experiments |
| StratifiedKFold CV | Class imbalance means random splits can be unrepresentative |
| Title extraction from Name | Titles encode age group and social class simultaneously |
| SMOTE in v5 | Titanic survivors are the minority class; improving recall matters |
| Companion survival feature | Family groups made survival decisions together (historical fact) |

---

## Interview Conclusion

This project documents the iterative machine learning workflow that data scientists use in practice: start with a working baseline, then improve it one hypothesis at a time. The progression from v1 to v5 demonstrates five distinct techniques: baseline modeling, domain-driven segmentation, feature engineering, group-level feature construction, and class balancing. The most interesting model is v4, which leverages the domain insight that Titanic passengers in the same group often survived or perished together — a real-world dependency that standard independent-sample models miss entirely. The risk in v4 is target leakage: you must only look up companion survival within the training set, never the test set. If I were continuing this, I would build a stacking ensemble using all five models' predictions as meta-features for a final logistic regression, which typically outperforms any single model.
