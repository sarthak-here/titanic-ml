"""
Model v1 - Simple Random Forest
Leaderboard score: 0.78229 (best score achieved)
Features: Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, Title, HasCabin
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(base, 'train.csv'))
test  = pd.read_csv(os.path.join(base, 'test.csv'))

def preprocess(df):
    df = df.copy()
    df['Sex']     = (df['Sex'] == 'female').astype(int)
    df['Age']     = df['Age'].fillna(df['Age'].median())
    df['Fare']    = df['Fare'].fillna(df['Fare'].median())
    df['Embarked']= df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title']   = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title']   = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    df['Title']   = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title']   = df['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}).fillna(0)
    df['HasCabin']= df['Cabin'].notna().astype(int)
    return df[['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','HasCabin']]

X      = preprocess(train)
y      = train['Survived']
X_test = preprocess(test)

model = RandomForestClassifier(
    n_estimators=200, max_depth=6,
    min_samples_split=4, min_samples_leaf=2,
    random_state=42
)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv     = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
model.fit(X, y)

print(f"CV: {cv.mean():.4f} +/- {cv.std():.4f}")

preds = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
submission.to_csv(os.path.join(base, 'submission_v1.csv'), index=False)
print("Saved: submission_v1.csv")
