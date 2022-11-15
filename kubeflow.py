# %%
import kfp
import kfp.dsl as dsl
from kfp import compiler
#from nodes import create_dataset, drop_unnecessary_columns
from kfp.v2.dsl import component, Dataset, OutputPath, InputPath, Input, Output
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
#import json
#import os 
# %%
@component(
    packages_to_install=['pandas', 'numpy', 'fsspec', 'gcsfs'],
)
def preprocessing_pipeline(path: str, dataset: OutputPath(Dataset)):
    
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path)

    columns = ['PassengerId']
    df.drop(columns, axis=1, inplace=True)

    mean = df['Age'].mean()
    std = df['Age'].std()
    total_nulls = df['Age'].isnull().sum()

    randon_age_range = np.random.randint(mean - std, mean + std, size=total_nulls)
    age_feat_slice = df['Age'].copy()
    age_feat_slice[np.isnan(age_feat_slice)] = randon_age_range

    df['Age'] = age_feat_slice
    df['Age'] = df['Age'].astype(int)

    common_val = 'S'

    df['Embarked'] = df['Embarked'].fillna(common_val)
    df['Fare'] = df['Fare'].fillna(0)
    df['Fare'] = df['Fare'].astype(int)

    df.to_csv(dataset, index=False)
# %%
@component(
    packages_to_install=['pandas','numpy','fsspec','gcsfs'],
)
def feature_engineering(path: InputPath(Dataset), train_dataset: OutputPath(Dataset)):
    
    import pandas as pd
    import re
    
    df = pd.read_csv(path)
    drop_cabin = False
    decks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df['Cabin'] = df['Cabin'].fillna('U0')
    df['Deck'] = df['Cabin'].apply(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].map(decks)
    df['Deck'] = df['Deck'].fillna(0)
    df['Deck'] = df['Deck'].astype(int)

    if drop_cabin:
        df.drop(['Cabin'], axis=1)

    drop_name = False
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other'
    )
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    if drop_name:
        df.drop(['Name'], axis=1, inplace=True)
    
    sex_dict = {"male": 0, "female": 1}
    df['Sex'] = df['Sex'].map(sex_dict)

    df['Relatives'] = df['SibSp'] + df['Parch']
    drop_features = False
    if drop_features:
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    columns = ['Cabin', 'Name', 'Ticket', 'SibSp', 'Parch']
    df.drop(columns,axis=1, inplace=True)

    encoded_ports = {'S': 0, 'C': 1, 'Q': 2}

    df['Embarked'] = df['Embarked'].map(encoded_ports)

    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare'] = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare'] = 4
    df.loc[df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int)

    df['Age'] = df['Age'].astype(int)

    df.loc[df['Age'] <= 11, 'Age'] = 0
    df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1
    df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2
    df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3
    df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4
    df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5
    df.loc[(df['Age'] > 40) & (df['Age'] <= 66), 'Age'] = 6
    df.loc[df['Age'] > 66, 'Age'] = 7

    titles_dic = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(titles_dic)
    df['Title'] = df['Title'].fillna(0)

    df['Age_Class'] = df['Age'] * df['Pclass']
    print(df.head())

    df.to_csv(train_dataset, index=False)
# %%
@component(
    packages_to_install=['pandas', 'numpy', 'fsspec', 'gcsfs', 'scikit-learn'],
)
def ml_pipeline(path: InputPath(Dataset)) -> float:
    
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv(path)

    x_train = df.drop(['Survived'], axis=1)
    y_train = df['Survived']

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    accuracy = model.score(x_train, y_train)
    print(accuracy)
    accuracy = round(accuracy * 100, 2)

    return accuracy
# %%
@dsl.pipeline(
    name = 'preprocessing-pipeline',
    description = 'Test to create the pipeline',
    pipeline_root = 'gs://titanic-challenge/artifacts'
)
def add_pipeline(path: str = 'gs://titanic-challenge/train.csv'):
    create_dataset = preprocessing_pipeline(path)
    create_feature = feature_engineering(create_dataset.outputs["dataset"])
    create_pipeline = ml_pipeline(path=create_feature.outputs["train_dataset"])
# %%
compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(pipeline_func=add_pipeline, package_path='pipeline.yaml')
# %%
