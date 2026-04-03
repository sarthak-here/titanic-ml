"""
Model v3 - Advanced Feature Engineering
Leaderboard score: 0.77751
New features over v1:
  - Title-based age imputation (smarter than global median)
  - Deck from cabin letter (E=best, U=unknown/worst)
  - One-hot Embarked (S/C/Q separately instead of ordinal)
  - TicketSize (companions on same ticket)
  - TrulyAlone (no family AND no ticket companions)
  - FarePerPerson (fare divided by ticket group size)
  - IsChild (age < 13, especially helpful for males)
  - BigFamily (family >= 5, near-certain death)
  - IsFreeTicket (fare=0 crew/staff, almost all died)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
all_d['TicketSize']    = all_d['Ticket'].map(tc)
all_d['IsFreeTicket']  = (all_d['Fare'] == 0).astype(int)
all_d['Fare']          = all_d['Fare'].fillna(all_d['Fare'].median()).replace(0, all_d['Fare'].median())
all_d['FarePerPerson'] = all_d['Fare'] / all_d['TicketSize']
all_d['Deck']          = all_d['Cabin'].str[0].map(
    {'E':5,'D':4,'A':3,'B':2,'F':2,'C':1,'G':1,'T':0}
).fillna(0).astype(int)
all_d['Embarked']   = all_d['Embarked'].fillna('S')
all_d['Emb_Q']      = (all_d['Embarked'] == 'Q').astype(int)
all_d['Emb_C']      = (all_d['Embarked'] == 'C').astype(int)
all_d['Sex']        = (all_d['Sex'] == 'female').astype(int)
all_d['FamilySize'] = all_d['SibSp'] + all_d['Parch'] + 1
all_d['IsAlone']    = (all_d['FamilySize'] == 1).astype(int)
all_d['TrulyAlone'] = ((all_d['FamilySize']==1) & (all_d['TicketSize']==1)).astype(int)
all_d['Title']      = all_d['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}).fillna(0)
all_d['HasCabin']   = all_d['Cabin'].notna().astype(int)
all_d['IsChild']    = (all_d['Age'] < 13).astype(int)
all_d['BigFamily']  = (all_d['FamilySize'] >= 5).astype(int)

features = ['Pclass','Sex','Age','FarePerPerson','Emb_Q','Emb_C',
            'FamilySize','IsAlone','TrulyAlone','Title','HasCabin',
            'Deck','IsChild','BigFamily','TicketSize','IsFreeTicket']

X      = all_d[features].iloc[:len(train)]
y      = train['Survived']
X_test = all_d[features].iloc[len(train):]

model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                min_samples_split=4, min_samples_leaf=2, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv  = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
model.fit(X, y)
print(f"CV: {cv.mean():.4f} +/- {cv.std():.4f}")

preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test_ids, 'Survived': preds}).to_csv(
    os.path.join(base, 'submission_v3.csv'), index=False)
print("Saved: submission_v3.csv")
