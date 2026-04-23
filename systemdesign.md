# Titanic ML - System Design

## What It Does
A Kaggle competition project solving Titanic survival prediction using five progressively
sophisticated ML approaches, each in its own module. Demonstrates iterative model
improvement from a baseline to advanced ensemble techniques.

---

## Architecture

```
titanic_model.py (orchestrator)
        |
   +----+----+----+----+----+
   |    |    |    |    |    |
   v1   v2   v3   v4   v5
simple segment feat  compa smote
  _rf  _ed   _eng  nion
```

---

## Dataset

```
Features: Pclass, Sex, Age (20% missing), SibSp, Parch,
          Fare, Embarked, Cabin (77% missing)
Target:   Survived (0 = died, 1 = survived)
Train:    891 rows   |   Test: 418 rows
```

---

## The Five Models

```
v1 - Simple Random Forest (baseline)
  Raw features -> median imputation -> LabelEncoding
  RandomForestClassifier(n_estimators=100)
  CV accuracy: ~80%  |  Goal: get something working fast

v2 - Segmented Models
  Insight: survival patterns differ by sex and class
  Train separate models per passenger segment:
    Women / Men in 1st class / Men in 2nd+3rd class
  +1-2% accuracy over v1

v3 - Feature Engineering
  New features:
    Title      = extracted from Name (Mr/Mrs/Miss/Master/Rare)
    FamilySize = SibSp + Parch + 1
    IsAlone    = 1 if FamilySize == 1
    Deck       = first letter of Cabin (A/B/.../Unknown)
    FareBand   = pd.qcut(Fare, 4) -> ordinal
    AgeBand    = pd.cut(Age, 5)   -> ordinal
  GradientBoostingClassifier
  +2-3% gain, especially Age handling

v4 - Companion Survival
  Insight: ticket-mates often survived or died together
  Feature: group survival rate from ticket grouping (train set only)
  XGBoost + companion feature
  Care: strict train/test separation to avoid target leakage

v5 - SMOTE
  Problem: slight class imbalance (62% perished)
  SMOTE oversamples minority class (survived=1)
  RandomForest + GridSearchCV tuning
  Focus: improve recall on survivors
```

---

## Data Flow (per model)

```
train.csv + test.csv
        |
  Preprocessing: impute Age (median by Title group),
                 Embarked (mode), Fare (median, test set)
        |
  Feature engineering (model-specific)
        |
  StratifiedKFold(5) cross-validation
        |
  Fit model -> CV accuracy + confusion matrix
        |
  Predict on test.csv -> submission.csv
```

---

## Key Design Decisions

| Decision                      | Reason                                             |
|-------------------------------|----------------------------------------------------|
| One file per model version    | Easy to compare; no shared mutable state           |
| StratifiedKFold CV            | Class imbalance makes random splits unreliable     |
| Title extraction from Name    | Encodes age group and social class simultaneously  |
| SMOTE in v5                   | Survivors are minority class; recall matters       |
| Companion survival feature    | Families made survival decisions together (history)|

---

## Interview Conclusion

This project documents the iterative ML workflow practitioners use: start with a working
baseline, then improve one hypothesis at a time. The v1-to-v5 progression demonstrates
five distinct techniques: baseline, domain-driven segmentation, feature engineering,
group-level feature construction, and class balancing. The most interesting model is v4:
it leverages the domain insight that Titanic passengers in the same group often survived
or perished together -- a real dependency that standard models miss. The risk in v4 is
target leakage: companion survival must only be looked up within the training set.
Next step: stack all five models with their predictions as meta-features for a final
logistic regression.
