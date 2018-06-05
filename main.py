import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier


# ============
# Inputs:
#     Training data split into X and y
#     Test data labeled as just test
#     X, y and test have to be arrays, .values or if you scale then that will make it an array
# ============

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)
folds = KFold(n_splits=5, shuffle=True, random_state=38)
sub_preds = np.zeros(test.shape[0])

for n_folds, (trn_idx, val_idx) in enumerate(folds.split(X)):
    trn_x, trn_y = X[trn_idx], y[trn_idx]
    val_x, val_y = X[val_idx], y[val_idx]
    
    clf = XGBClassifier(
        max_depth=8
        , subsample=.8
        , colsample_bytree=.8
        , reg_alpha=.1
        , reg_lambda=.1
)
    
    clf.fit(trn_x, trn_y
           , eval_set=[(trn_x, trn_y),(val_x, val_y)]
           , eval_metric='auc'
           , verbose=100
           , early_stopping_rounds=100
           )
# Is taking the mean of all predictions better than just predicting on test set?
    sub_preds += clf.predict_proba(test)[:,1] / folds.n_splits

test_preds = sub_preds
pred_df = pd.DataFrame()
pred_df['SK_ID_CURR'] = test['SK_ID_CURR']
pred_df['TARGET'] = test_preds
pred_df.to_csv('xgb_main.csv', index=False)