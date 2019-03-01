import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

data_train = pd.read_csv("g:/Project/python/kaggle/Titanic/train.csv")
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'NameLength']

# data pretreatment
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_train.loc[data_train['Sex'] == 'male', 'Sex'] = 0
data_train.loc[data_train['Sex'] == 'female', 'Sex'] = 1
# data_train['Sex'].unique()
data_train['Embarked'] = data_train['Embarked'].fillna('S')
data_train.loc[data_train['Embarked'] == "S", "Embarked"] = 0
data_train.loc[data_train['Embarked'] == "C", "Embarked"] = 1
data_train.loc[data_train['Embarked'] == "Q", "Embarked"] = 2

# feature extraction
# family size
data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch']
# the length of name
data_train['NameLength'] = data_train['Name'].apply(lambda x: len(x))
# add title
titles = data_train['Name'].apply(get_title)
# print(pd.value_counts(titles))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Sir": 9, "Don": 10, "Mme": 11, "Capt": 12, "Ms": 13, "Lady": 14, "Countess": 15, "Jonkheer": 16}
for k, v in title_mapping.items():
    titles[titles == k] = v

# print(pd.value_counts(titles))
data_train['Title'] = titles

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
kf = cross_validation.KFold(data_train.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, data_train[predictors], data_train['Survived'], cv=kf)
print(scores.mean())

# feature select
selector = SelectKBest(f_classif, k=5)
selector.fit(data_train[predictors], data_train['Survived'])
scores = -np.log10(selector.pvalues_)

# plot the scores
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

predictors = ['Pclass', 'Sex', 'Fare', 'Title', 'NameLength']
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
kf = cross_validation.KFold(data_train.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, data_train[predictors], data_train['Survived'], cv=kf)
print(scores.mean())


# 集成算法
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ['Pclass', 'Sex', 'Fare', 'FamilySize', 'Title', 'Age', 'Embarked']],
    [LogisticRegression(random_state=1), ['Pclass', 'Sex', 'Fare', 'FamilySize', 'Title', 'Age', 'Embarked']]
]
kf = cross_validation.KFold(data_train.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_target = data_train['Survived'].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(data_train[predictors].iloc[train, :], train_target)
        test_predictions = alg.predict_proba(data_train[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
# accuracy = sum(predictions[predictions == data_train['Survived']]) / len(predictions)
summ = 0
survived_list = list(data_train['Survived'])
for i in range(len(predictions)):
    if predictions[i] == survived_list[i]:
        summ += 1
accuracy = summ / len(predictions)
print(accuracy)
