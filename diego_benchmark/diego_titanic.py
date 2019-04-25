#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diego_benchmark/diego_titanic.py was created on 2019/04/06.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
from sklearn.preprocessing import LabelEncoder
from base_benchmark import simple_diego
import pandas as pd
import numpy as np





train_df_raw = pd.read_csv('data/titanic/train.csv')
test_df_raw = pd.read_csv('data/titanic/test.csv')

def preprocess_data(df):
    
    processed_df = df
        
    ########## Deal with missing values ##########
    
    # As we saw before, the two missing values for embarked columns can be replaced by 'C' (Cherbourg)
    processed_df['Embarked'].fillna('C', inplace=True)
    
    # We replace missing ages by the mean age of passengers who belong to the same group of class/sex/family
    processed_df['Age'] = processed_df.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))
    processed_df['Age'] = processed_df.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
    processed_df['Age'] = processed_df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
    
    # We replace the only missing fare value for test processed_df and the missing values of the cabin column
    processed_df['Fare'] = processed_df['Fare'].interpolate()
    processed_df['Cabin'].fillna('U', inplace=True)
    
    ########## Feature engineering on columns ##########
    
    # Create a Title column from name column
    processed_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in train_df_raw['Name']), index=train_df_raw.index)
    processed_df['Title'] = processed_df['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    processed_df['Title'] = processed_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    processed_df['Title'] = processed_df['Title'].replace('Mme', 'Mrs')
 
    
    # Create a Family Size, Is Alone, Child and Mother columns
    processed_df['FamilySize'] = processed_df['SibSp'] + processed_df['Parch'] + 1
    processed_df['FamilySize'][processed_df['FamilySize'].between(1, 5, inclusive=False)] = 2
    processed_df['FamilySize'][processed_df['FamilySize']>5] = 3
    processed_df['IsAlone'] = np.where(processed_df['FamilySize']!=1, 0, 1)
    processed_df['IsChild'] = processed_df['Age'] < 18
    processed_df['IsChild'] = processed_df['IsChild'].astype(int)
    
    # Modification of cabin column to keep only the letter contained corresponding to the deck of the boat
    processed_df['Cabin'] = processed_df['Cabin'].str[:1]
    processed_df['Cabin'] = processed_df['Cabin'].map({cabin: p for p, cabin in enumerate(set(cab for cab in processed_df['Cabin']))})
    
    
    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    processed_df['FareBin'] = pd.qcut(processed_df['Fare'], 4)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    processed_df['AgeBin'] = pd.cut(processed_df['Age'].astype(int), 5)
    
    label = LabelEncoder()

    # Converting. 
    processed_df['Sex_Code'] = label.fit_transform(processed_df['Sex'])
    processed_df['Embarked_Code'] = label.fit_transform(processed_df['Embarked'])
    processed_df['Title_Code'] = label.fit_transform(processed_df['Title'])
    processed_df['AgeBin_Code'] = label.fit_transform(processed_df['AgeBin'])
    processed_df['FareBin_Code'] = label.fit_transform(processed_df['FareBin'])
    
    dummy_cols = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    data1_dummy = pd.get_dummies(processed_df[dummy_cols])
    print(data1_dummy.columns)
    data1_dummy['PassengerId'] = processed_df['PassengerId']
    
    # converting
    processed_df['Title'] = processed_df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    
    # Filling Age missing values with mean age of passengers who have the same title
    processed_df['Age'] = processed_df.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    # Transform categorical variables to numeric variables
    processed_df['Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})
    processed_df['Embarked'] = processed_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    processed_df = pd.merge(processed_df, data1_dummy, on=['PassengerId'])
    # Create a ticket survivor column which is set to 1 if an other passenger with the same ticket survived and 0 else
    # Note : this implementation is ugly and unefficient, if sombody found a way to do it easily with pandas (it must be a way), please comment the kernel with your solution !
    processed_df['TicketSurvivor'] = pd.Series(0, index=processed_df.index)
    tickets = processed_df['Ticket'].value_counts().to_dict()
    for t, occ in tickets.items():
        if occ != 1:
            table = train_df_raw['Survived'][train_df_raw['Ticket'] == t]
            if sum(table) != 0:
                processed_df['TicketSurvivor'][processed_df['Ticket'] == t] = 1
    
    # These two columns are not useful anymore
    processed_df = processed_df.drop(['Name', 'Ticket', 'PassengerId', 'AgeBin', 'FareBin'], 1)    
    
    return processed_df

def main():


    train_df = train_df_raw.copy()
    X = train_df.drop(['Survived'], 1)
    Y = train_df['Survived']
    X = preprocess_data(X)
    final_clf = simple_diego(X, Y)

    X_test = preprocess_data(test_df_raw)
    y_test = final_clf.predict(X_test)
    submission = pd.DataFrame()
    submission['PassengerId'] = test_df_raw['PassengerId']
    submission['Survived'] = y_test
    submission.to_csv('data/titanic/titanic_submission1.csv', encoding='utf8',index=False)

if __name__ == "__main__":
    main()