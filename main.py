# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, ndcg_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
import itertools
from xgboost import XGBClassifier
import warnings
import argparse
warnings.filterwarnings("ignore")

import utils
import neural_network
import rank

def load_dataset():
    raw_dataset = pd.read_csv('CollegeBasketballPlayers2009-2021.csv', low_memory=False)

    new_dataset = raw_dataset.loc[raw_dataset['pick'] >= 1]
    trainset = new_dataset.loc[new_dataset['year'] < 2021]
    testset = new_dataset.loc[raw_dataset['year'] == 2021]
    return trainset, testset

def preprocess(trainset, testset):
    features_basic = ['treb', 'ast', 'stl', 'blk', 'pts', 'yr']
    features_advanced = ['eFG', 'TS_per', 'FT_per', 'twoP_per', 'TP_per', 'ast/tov', 
                     'obpm', 'dbpm', 'oreb', 'dreb', 'TO_per', 'ORB_per', 'DRB_per',
                     'AST_per', 'blk_per', 'stl_per', 'Min_per']
    trainset['yr'] = trainset['yr'].rank(method='dense', ascending=True).astype(int)
    testset['yr'] = testset['yr'].rank(method='dense', ascending=True).astype(int)
    trainset['ast/tov'] = trainset['ast/tov'].fillna(trainset['ast/tov'].value_counts().index[1])
    X_train = np.asarray(trainset[features_basic + features_advanced])
    X_test = np.asarray(testset[features_basic + features_advanced])
    y_train_pick = np.asarray(trainset.pick)
    y_train_year = np.asarray(trainset.year)
    y_test = np.asarray(testset.pick)
    return X_train, y_train_pick, y_train_year, X_test, y_test

def use_regression_model(X_train, y_train, X_test, y_test):
    ball_model = RandomForestRegressor()
    ball_model.fit(X_train, y_train)
    y_pred = ball_model.predict(X_test)
    y_pred = y_pred.argsort().argsort()
    y_test = y_test.argsort().argsort()
    mae = mean_absolute_error(y_pred, y_test)
    print("Mean Absolute Error: {:,.5f}".format(mae))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBA Draft Prediction Program')
    args = parser.parse_args()
    #TODO

    trainset, testset = load_dataset()
    X_train, y_train_pick, y_train_year, X_test, y_test = preprocess(trainset, testset)
    y_train = np.c_[y_train_pick, y_train_year]
    #rank_pred = neural_network.use_neural_network(X_train, y_train, X_test, y_test)
    rank_pred = rank.use_rank_model(X_train, y_train, X_test, y_test)

    df = pd.DataFrame(data={'Name': testset['player_name'].values,
                        'Rank': np.array([item + 1 for item in rank_pred])})
    df.to_csv('Rank.csv') 
    print('Best NDCG predictions saved to Rank.csv.')

