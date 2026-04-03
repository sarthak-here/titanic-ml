"""
Model v2 - Segmented Model
Leaderboard score: 0.77751
Approach: Rule-based for high-certainty groups, RF model for uncertain group
  - female 1st/2nd class -> predict survived (94.7% accurate in training)
  - male 2nd/3rd class   -> predict died    (85.9% accurate in training)
  - uncertain (female 3rd + male 1st) -> RandomForest
Insight: Breaking dataset into segments based on known survival patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(base, 'train.csv'))
test  = pd.read_csv(os.path.join(base, 'test.csv'))
test_ids = test['PassengerId'].copy()

all_data = pd.concat([train.drop('Survived', axis=1), test], ignore_index=True)

all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
)
all_data['Title'] = all_data['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
all_data['Age'] = all_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone']    = (all_data['FamilySize'] == 1).astype(int)
all_data['Fare']       = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['HasCabin']   = all_data['Cabin'].notna().astype(int)
all_data['Embarked']   = all_data['Embarked'].fillna('S')

train_p = all_data.iloc[:len(train)].copy()
test_p  = all_data.iloc[len(train):].copy()
train_p['Survived'] = train['Survived'].values

def get_segment(df):
    s = pd.Series('uncertain', index=df.index)
    s[(df['Sex']=='female') & (df['Pclass'].isin([1,2]))] = 'female_1st_2nd'
    s[(df['Sex']=='male')   & (df['Pclass'].isin([2,3]))] = 'male_2nd_3rd'
    return s

train_p['Segment'] = get_segment(train_p)
test_p['Segment']  = get_segment(test_p)

unc_train = train_p[train_p['Segment']=='uncertain'].copy()
unc_test  = test_p[test_p['Segment']=='uncertain'].copy()

le = LabelEncoder()
le.fit(pd.concat([unc_train['Title'], unc_test['Title']]))
unc_train['TitleEnc'] = le.transform(unc_train['Title'])
unc_test['TitleEnc']  = le.transform(unc_test['Title'])

features = ['Age','Fare','FamilySize','IsAlone','TitleEnc','SibSp','Parch']
model = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5,
                                max_features='sqrt', random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv  = cross_val_score(model, unc_train[features], unc_train['Survived'], cv=skf, scoring='accuracy')
print(f"Uncertain group CV: {cv.mean():.4f} +/- {cv.std():.4f}")

model.fit(unc_train[features], unc_train['Survived'])
unc_preds = model.predict(unc_test[features])

results = pd.DataFrame({'PassengerId': test_ids, 'Survived': -1})
f_idx  = test_p[test_p['Segment']=='female_1st_2nd'].index - test_p.index[0]
m_idx  = test_p[test_p['Segment']=='male_2nd_3rd'].index   - test_p.index[0]
un_idx = test_p[test_p['Segment']=='uncertain'].index      - test_p.index[0]
results.iloc[f_idx,  1] = 1
results.iloc[m_idx,  1] = 0
results.iloc[un_idx, 1] = unc_preds

results.to_csv(os.path.join(base, 'submission_v2.csv'), index=False)
print("Saved: submission_v2.csv")
