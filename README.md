# Titanic - Machine Learning from Disaster

Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

**Best leaderboard score: 0.78229**

---

## Setup

```bash
pip install -r requirements.txt
```

Download the data from Kaggle and place `train.csv` and `test.csv` in the root folder:

```bash
kaggle competitions download -c titanic
unzip titanic.zip
```

---

## Models

### v1 — Simple Random Forest (`models/v1_simple_rf.py`)
**Score: 0.78229**

Baseline model. Simple feature engineering with global median age imputation.

Features: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `FamilySize`, `IsAlone`, `Title`, `HasCabin`

---

### v2 — Segmented Model (`models/v2_segmented.py`)
**Score: 0.77751**

Splits passengers into three segments based on survival certainty:
- `female 1st/2nd class` → rule: predict **survived** (94.7% accurate)
- `male 2nd/3rd class` → rule: predict **died** (85.9% accurate)
- `uncertain` (female 3rd + male 1st) → **RandomForest**

Key insight from training data survival rates by sex and class.

---

### v3 — Advanced Feature Engineering (`models/v3_feature_engineering.py`)
**Score: 0.77751**

All features from v1 plus:
- **Title-based age imputation** — fill missing ages using median per title group (Masters are young, Mrs are older)
- **Deck** — cabin letter mapped by survival rate (E=best, U=unknown/worst)
- **One-hot Embarked** — S/C/Q as separate binary flags (Southampton had lower survival)
- **TicketSize** — passengers sharing a ticket (companions) had 51% survival vs 27% alone
- **TrulyAlone** — no family AND no ticket companions
- **FarePerPerson** — fare divided by ticket group size (removes suite fare distortion)
- **IsChild** — age < 13 (children had priority especially on 2nd class lifeboats)
- **BigFamily** — family size ≥ 5 (near-certain death, all large families perished)
- **IsFreeTicket** — fare = 0 crew/staff (1/15 survived)

---

### v4 — Companion Survival Rate (`models/v4_companion_survival.py`) ⭐ BEST
**Score: 0.78229**

All features from v3 plus the most powerful new signal:

**CompanionSurvRate** — for each passenger, what fraction of their ticket companions (in training set) survived?
- Companions all survived → **70.9%** chance the passenger survived
- Companions all died → **22.5%** chance
- 152 test passengers (36%) have training companions

Computed with **leave-one-out** on training data to avoid data leakage.

---

### v5 — SMOTE Augmentation (`models/v5_smote.py`)
**Score: ~0.77 (did not improve)**

Experiment with synthetic data generation (SMOTE) for underrepresented groups:
- Male 2nd class: only 17 survivors (very imbalanced)
- Male 1st class: only 45 survivors

Outliers excluded from SMOTE pool before generating synthetic rows:
- Fare = 0 (crew/staff)
- Age ≥ 65
- FamilySize ≥ 5
- Rare titles

**Lesson:** SMOTE on small tabular datasets with inherent randomness tends to overfit.
The synthetic passengers don't represent real test patterns.

---

## Key Findings

| Group | Train survival rate | Approach |
|---|---|---|
| Female 1st class | 96.8% | Predict survived |
| Female 2nd class | 92.1% | Predict survived |
| Male 2nd/3rd class | 13-16% | Predict died |
| Female 3rd class | 50.0% | Hard to predict — genuine randomness |
| Male 1st class | 36.9% | Model needed (HasCabin, Deck help) |

**The honest ceiling for this dataset with standard ML is ~0.78-0.82.**
Beyond that requires either data leakage (historical records) or overfitting the public leaderboard.

---

## Running the best model

```bash
python models/v4_companion_survival.py
```

This generates `submission_v4.csv` ready for Kaggle submission.
