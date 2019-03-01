import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

data_train = pd.read_csv("g:/Project/python/kaggle/Titanic/train.csv")
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# data_train
# data_train.describe()

# 数据预处理
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_train.loc[data_train['Sex'] == 'male', 'Sex'] = 0
data_train.loc[data_train['Sex'] == 'female', 'Sex'] = 1
# data_train['Sex'].unique()
data_train['Embarked'] = data_train['Embarked'].fillna('S')
data_train.loc[data_train['Embarked'] == "S", "Embarked"] = 0
data_train.loc[data_train['Embarked'] == "C", "Embarked"] = 1
data_train.loc[data_train['Embarked'] == "Q", "Embarked"] = 2

alg = LinearRegression()
kf = KFold(data_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (data_train[predictors].iloc[train, :])
    train_target = data_train['Survived'].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(data_train[predictors].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

# 不知道问题出在哪里
# accuracy = sum(predictions[predictions == data_train['Survived']]) / len(predictions)
# print(accuracy)

summ = 0
survived_list = list(data_train['Survived'])
for i in range(len(predictions)):
    if predictions[i] == survived_list[i]:
        summ += 1
accuracy = summ / len(predictions)
print(accuracy)

# 逻辑回归
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, data_train[predictors], data_train['Survived'], cv=3)
print(scores.mean())

