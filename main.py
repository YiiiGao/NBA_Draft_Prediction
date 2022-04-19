# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, ndcg_score

import argparse
import warnings
warnings.filterwarnings("ignore")

import utils
import neural_network
import rank

def load_dataset(year):
    raw_dataset = pd.read_csv('CollegeBasketballPlayers2009-2021.csv', low_memory=False)

    new_dataset = raw_dataset.loc[raw_dataset['pick'] >= 1]
    trainset = new_dataset.loc[new_dataset['year'] != year]
    testset = new_dataset.loc[raw_dataset['year'] == year]
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
    return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBA Draft Prediction Program')
    parser.add_argument('--method', type=str, help='Chooose the learning method you like between regression and ranking', 
                                    default='ranking', choices=['regression', 'ranking'])
    parser.add_argument('--year', type=int, help='Choose a year from 2009 to 2021 as the testset',
                                    default=2021)
    parser.add_argument('--use_nn', type=bool, help='Choose whether to use the neural network',
                                    default=False)
    args = parser.parse_args()
    
    if args.year < 2009 or args.year > 2021:
        raise ValueError('Unsupported year number!')
    if args.method == 'Regression' and args.use_nn == True:
        raise ValueError('Cannot use the neural network for regression!')

    trainset, testset = load_dataset(args.year)
    X_train, y_train_pick, y_train_year, X_test, y_test = preprocess(trainset, testset)
    y_train = np.c_[y_train_pick, y_train_year]
    if args.method == 'ranking':
        rank_pred = rank.use_rank_model(X_train, y_train, X_test, y_test)
        if args.use_nn:
            rank_nn = neural_network.use_neural_network(X_train, y_train, X_test, y_test)
            df_nn = pd.DataFrame(data={'Name': testset['player_name'].values,
                        'Rank': np.array([item + 1 for item in rank_pred])})
            df_nn.to_csv(f'Rank_{args.year}_nn.csv') 
            print(f'Neural Network Predictions saved to Rank_{args.year}_nn.csv.') 

    elif args.method == 'regression':
        rank_pred = use_regression_model(X_train, y_train_pick, X_test, y_test)
    else:
        raise ValueError('Unsupported method type!')

    df = pd.DataFrame(data={'Name': testset['player_name'].values,
                        'Rank': np.array([item + 1 for item in rank_pred])})
    df.to_csv(f'Rank_{args.year}.csv') 
    print(f'Best NDCG predictions saved to Rank_{args.year}.csv.')

