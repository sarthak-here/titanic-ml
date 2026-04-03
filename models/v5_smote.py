"""
Model v5 - SMOTE Data Augmentation
Leaderboard score: ~0.77 (did not beat v4)
Experiment: Synthetic Minority Over-sampling Technique on scarce groups
  - Male 2nd class has only 17 survivors (very imbalanced)
  - Male 1st class has 45 survivors (imbalanced)
  - Outliers removed before SMOTE to avoid bad interpolation:
      * Fare=0 crew/staff
      * Age >= 65 (old passengers, nearly all died)
      * FamilySize >= 5 (large families, nearly all died)
      * Rare titles (Rev, Col etc — all died)
Lesson learned: SMOTE on tabular data with inherent randomness tends to overfit.
The synthetic passengers don't represent reality well enough to help on test data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(base, 'train.csv'))
test  = pd.read_csv(os.path.join(base, 'test.csv'))
test_ids = test['PassengerId'].copy()

all_d = pd.concat([train.drop('Survived', axis=1), test], ignore_index=True)
all_d['Title'] = all_d['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_d['Title'] = all_d['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
)
all_d['Title'] = all_d['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
all_d['Age']   = all_d.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

tc = all_d['Ticket'].value_counts()
all_d['TicketSize'] = all_d['Ticket'].map(tc)

ts = train.groupby('Ticket')['Survived'].mean().rename('CompanionSurvRate')
all_d = all_d.merge(ts, on='Ticket', how='left')
gs = train['Survived'].mean()
cr = []
for i in range(len(train)):
    same = train[train['Ticket']==train['Ticket'].iloc[i]]
    others = same[same.index!=i]
    cr.append(others['Survived'].mean() if len(others)>0 else np.nan)
cr = np.where(np.isnan(cr), gs, cr)

all_d['IsFreeTicket']  = (all_d['Fare']==0).astype(int)
all_d['Fare']          = all_d['Fare'].fillna(all_d['Fare'].median()).replace(0, all_d['Fare'].median())
all_d['FarePerPerson'] = all_d['Fare'] / all_d['TicketSize']
all_d['Deck']          = all_d['Cabin'].str[0].map(
    {'E':5,'D':4,'A':3,'B':2,'F':2,'C':1,'G':1,'T':0}
).fillna(0).astype(int)
all_d['Embarked']   = all_d['Embarked'].fillna('S')
all_d['Emb_Q']      = (all_d['Embarked']=='Q').astype(int)
all_d['Emb_C']      = (all_d['Embarked']=='C').astype(int)
all_d['Sex']        = (all_d['Sex']=='female').astype(int)
all_d['FamilySize'] = all_d['SibSp'] + all_d['Parch'] + 1
all_d['IsAlone']    = (all_d['FamilySize']==1).astype(int)
all_d['TrulyAlone'] = ((all_d['FamilySize']==1)&(all_d['TicketSize']==1)).astype(int)
all_d['Title']      = all_d['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}).fillna(0)
all_d['HasCabin']   = all_d['Cabin'].notna().astype(int)
all_d['IsChild']    = (all_d['Age']<13).astype(int)
all_d['BigFamily']  = (all_d['FamilySize']>=5).astype(int)

features = ['Pclass','Sex','Age','FarePerPerson','Emb_Q','Emb_C',
            'FamilySize','IsAlone','TrulyAlone','Title','HasCabin',
            'Deck','IsChild','BigFamily','TicketSize','IsFreeTicket','CompanionSurvRate']

X = all_d[features].iloc[:len(train)].copy()
X['CompanionSurvRate'] = cr
y = train['Survived'].values
X_test = all_d[features].iloc[len(train):].copy()
X_test['CompanionSurvRate'] = all_d['CompanionSurvRate'].iloc[len(train):].fillna(gs).values

# outliers to exclude from SMOTE pool
orig_fare  = train['Fare'].fillna(0).values
orig_title = all_d['Title'].iloc[:len(train)].values
outlier_mask = (
    (orig_fare == 0) |
    (all_d['Age'].iloc[:len(train)].values >= 65) |
    (all_d['FamilySize'].iloc[:len(train)].values >= 5) |
    (orig_title == 5)
)

smote_groups = {
    'male_1st': (train['Sex']=='male') & (train['Pclass']==1),
    'male_2nd': (train['Sex']=='male') & (train['Pclass']==2),
}

smote = SMOTE(random_state=42, k_neighbors=5)
X_extras, y_extras = [], []
for name, mask in smote_groups.items():
    clean = mask.values & ~outlier_mask
    Xg, yg = X[clean], y[clean]
    if yg.sum() >= 2 and (yg==0).sum() >= 2:
        Xr, yr = smote.fit_resample(Xg, yg)
        X_extras.append(pd.DataFrame(Xr[len(Xg):], columns=X.columns))
        y_extras.append(yr[len(yg):])
        print(f"{name}: added {len(Xr)-len(Xg)} synthetic rows")

X_aug = pd.concat([X] + X_extras, ignore_index=True)
y_aug = np.concatenate([y] + y_extras)

rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                             min_samples_split=4, min_samples_leaf=2, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for ti, vi in skf.split(X, y):
    X_tr = X.iloc[ti].copy().reset_index(drop=True)
    y_tr = y[ti].copy()
    fold_extras_X, fold_extras_y = [], []
    for name, mask in smote_groups.items():
        clean_fold = mask.values[ti] & ~outlier_mask[ti]
        Xg, yg = X_tr[clean_fold], y_tr[clean_fold]
        if yg.sum() >= 2 and (yg==0).sum() >= 2:
            Xr, yr = smote.fit_resample(Xg, yg)
            fold_extras_X.append(pd.DataFrame(Xr[len(Xg):], columns=X.columns))
            fold_extras_y.append(yr[len(yg):])
    if fold_extras_X:
        X_fold = pd.concat([X_tr] + fold_extras_X, ignore_index=True)
        y_fold = np.concatenate([y_tr] + fold_extras_y)
    else:
        X_fold, y_fold = X_tr, y_tr
    rf.fit(X_fold, y_fold)
    cv_scores.append((rf.predict(X.iloc[vi])==y[vi]).mean())

print(f"CV: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
rf.fit(X_aug, y_aug)

preds = rf.predict(X_test)
pd.DataFrame({'PassengerId': test_ids, 'Survived': preds}).to_csv(
    os.path.join(base, 'submission_v5.csv'), index=False)
print("Saved: submission_v5.csv")
