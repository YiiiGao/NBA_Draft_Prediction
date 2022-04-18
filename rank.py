import numpy as np
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

def all_models():
    model_list = []
    xgb = XGBClassifier()
    model_list.append(xgb)
    rf = RandomForestClassifier()
    model_list.append(rf)
    dt = DecisionTreeClassifier()
    model_list.append(dt)
    kn = KNeighborsClassifier()
    model_list.append(kn)
    return model_list
    

def use_rank_model(X_train, y_train, X_test, y_test):
    start = time.time()
    X_new, y_new = utils.transform_pairwise(X_train, y_train)
    y_new = np.where(y_new == -1, 0, y_new)
    X_test_new, y_test_new = utils.transform_pairwise(X_test, y_test)
    y_test_new = np.where(y_test_new == -1, 0, y_test_new)
    best_acc = 0
    best_ndcg = 0
    for rank_model in all_models():
        rank_model.fit(X_new, y_new)
        y_pred = rank_model.predict(X_test_new)
        accuracy = accuracy_score(y_test_new, y_pred)
        if accuracy > best_acc:
            best_acc = accuracy
        rank_pred, score = utils.calc_ndcg(y_pred, y_test)
        if score > best_ndcg:
            best_ndcg = score
    print('Best pairwise accuracy: {:,.5f}'.format(best_acc))
    print('Best NDCG score: {:,.5f}'.format(best_ndcg))
    end = time.time()
    print('Runtime: {:,.5f}s'.format(end - start))
    return rank_pred