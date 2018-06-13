
# coding: utf-8

# In[522]:

import pandas as pd
import numpy as np
pd.options.display.max_rows = 2000
pd.options.display.max_columns = 2000
from sklearn.utils import resample

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# Read data:

# In[523]:

path = '../../data/'


# In[524]:

df_train = pd.read_csv(path+'application_train.csv')
# df_cc_bal = pd.read_csv(path+'credit_card_balance.csv')
df_pos = pd.read_csv(path+'POS_CASH_balance.csv')


# In[525]:

df_train.shape


# In[526]:

df_train.groupby(by='TARGET')['SK_ID_CURR'].nunique()


# In[527]:

df_majority = df_train[df_train.TARGET==0]
df_minority = df_train[df_train.TARGET==1]


# In[528]:

# downsample majority class
df_majority_downsampled = resample(df_majority,replace=True,                                  n_samples=len(df_minority),                                  random_state=23)

df_train = pd.concat([df_majority_downsampled,df_minority])


# In[529]:

# # upsample minority class
# df_majority_upsampled = resample(df_minority,replace=True,\
#                                   n_samples=len(df_majority),\
#                                   random_state=23)

# df_train = pd.concat([df_majority_upsampled,df_majority])


# In[ ]:




# In[530]:

df_train.shape


# In[531]:

df_train.head()


# Feature engineering:

# Before any feature engineering, ROCAUC:

# In[532]:

0.75487777950760337


# In[533]:

AMT_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']

for i in range(len(AMT_cols)):
    for j in range(len(AMT_cols)):
        if i != j:
            col_1 = AMT_cols[i]
            col_2 = AMT_cols[j]
            df_train[col_1+'_OVER_'+col_2] = 1.0 * df_train[col_1] / df_train[col_2]


# ROCAUC:

# In[534]:

0.76034363338930977


# In[535]:

control_for_age_cols = ['DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']

for col in control_for_age_cols:
    df_train[col+'_OVER_DAYS_BIRTH'] = 1.0 * df_train[col] / df_train.DAYS_BIRTH


# ROCAUC:

# In[536]:

0.76111507043858218


# In[537]:

df_train['CNT_FAM_MEMBERS_OVER_DAYS_BIRTH'] = 1.0 * df_train.CNT_FAM_MEMBERS / (-df_train.DAYS_BIRTH)


# ROCAUC:

# In[538]:

0.76153678901265509


# Adding AdaBoost to Voting Classifier, ROCAUC:

# In[539]:

0.76158009812369776


# In[540]:

# df_cc_bal.head()


# In[541]:

# df_cc_bal = df_cc_bal[[x for x in df_cc_bal.columns if x not in ['SK_ID_PREV']]]


# In[542]:

# df_cc_bal_active = df_cc_bal.loc[df_cc_bal.NAME_CONTRACT_STATUS == 'Active', :]


# In[543]:

# df_cc_bal_active_mean = df_cc_bal_active.groupby(by='SK_ID_CURR').mean()
# df_cc_bal_active_median = df_cc_bal_active.groupby(by='SK_ID_CURR').median()
# df_cc_bal_active_std = df_cc_bal_active.groupby(by='SK_ID_CURR').std()


# In[544]:

# df_cc_bal_active_mean.shape, df_cc_bal_active_median.shape, df_cc_bal_active_std.shape


# In[545]:

# df_cc_bal_active_agg = pd.concat([df_cc_bal_active_mean,df_cc_bal_active_median,df_cc_bal_active_std])


# In[546]:

# df_cc_bal_active_agg.reset_index(inplace=True)


# In[547]:

# df_all = df_train.join(other=df_cc_bal_active_agg,how='left',on='SK_ID_CURR',rsuffix='_copy')
# del(df_all['SK_ID_CURR_copy'])


# ROCAUC:

# In[548]:

0.7613472727048014


# In[549]:

df_pos.head()


# In[550]:

df_pos = df_pos[[x for x in df_pos.columns if x not in ['SK_ID_PREV']]]


# In[551]:

df_pos_median = df_pos.groupby(by='SK_ID_CURR').median()
df_pos_std = df_pos.groupby(by='SK_ID_CURR').std()

df_pos_agg = pd.concat([df_pos_median,df_pos_std])


# In[552]:

df_pos.reset_index(inplace=True)


# In[553]:

df_all = df_train.join(other=df_pos,how='left',on='SK_ID_CURR',rsuffix='_copy')
df_all[[x for x in df_all.columns if x != 'SK_ID_CURR_copy']]


# In[554]:

df_all.shape


# ROCAUC (admittedly i changed filling na to from 0 to med here)

# Model:

# In[555]:

X = df_all[[x for x in df_all.columns if x not in ['SK_ID_CURR','TARGET']]]
y = df_all['TARGET']


# In[556]:

X.shape


# In[557]:

X = X.fillna('med')


# In[558]:

X = pd.get_dummies(X, dummy_na = True, drop_first = True)


# In[559]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,                                                    random_state = 23)


# Voting Classifier:

# In[560]:

# # kfold = model_selection.KFold(n_splits=10, random_state=23)

# estimators = []
# model1 = GradientBoostingClassifier(random_state = 23)
# estimators.append(('one', model1))
# model2 = AdaBoostClassifier(random_state = 23)

# clf = VotingClassifier(estimators, voting = 'soft')


# In[ ]:

clf = GradientBoostingClassifier(random_state = 23)


# In[ ]:

y_pred = clf.fit(X_train, y_train).predict_proba(X_test)


# In[ ]:

y_pred = [y[1] for y in y_pred]


# In[ ]:

roc_auc_score(y_test,y_pred)


# In[ ]:

X.shape


# In[ ]:



