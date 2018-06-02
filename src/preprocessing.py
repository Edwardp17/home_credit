# Import libraries
import os
import gc
gc.enable()
import random
import numpy as np 
import pandas as pd

DATA_PATH = '../input/'
DATASET_FILENAMES = []


class Dataset:


    def __init__(self, filename, join_key = None, exclude_vars = [],\
     cat_vars = [], num_vars = [], binary_vars = []):

        self.name = filename.split('.')[0]
        self.filename = filename
        self.join_key = join_key

        self.exclude_vars = exclude_vars
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.binary_vars = binary_vars

        self.cat_features = None
        self.num_features = None
        self.binary_features = None

        self.data = None
        self.features = None
        self.target = None


    def read_data(self):

        self.data = pd.read_csv(DATA_PATH + self.filename)


    def discover_binary_vars(self):
        
        if self.data = None:

            raise Exception('Data cannot be None for binary vars to be discovered.')
        
        for col in self.data.columns:

            if len(self.data[col].unique()) == 2:

                self.binary_vars.append(col)


    def transform_binary_features(self):

        # redundant statements like this can be put in a decorator
        if self.data = None:

            raise Exception('Data cannot be None for binary vars to be discovered.')

        df_binary_features = self.data[self.binary_features]

        for col in self.binary_vars:

            s_binary_feature = self.data[col]
            unique_values = list(s_binary_feature.unique())

            if set(unique_values) != set([0,1]):
                
                standardization_map = dict(zip(unique_values),[0,1])
                s_binary_feature = [standardization_map[x] for x in list(s_binary_feature)]
                df_binary_features[col] = s_binary_feature

        self.binary_features = df_binary_features
        

class Preprocessor:


    def __init__(self):
        
        self.datasets = None

        self.df_application_train_features = None


    def read_all_data(self, dataset_filenames=[]):

        datasets = []

        for d in dataset_filenames:

            dataset = Dataset(filename = d).read_data()

        self.datasets = datasets

    # ================
    # application_train
    # ================


    def get_application_train_features(self, application_train_name = 'application_train', include_join_key = True):

        df_application_train = None
        for df in self.datasets:

            if df.name == application_train_name:

                df_application_train = df
        
        if df_application_train = None:

            raise Exception('Could not find application_train in Preprocessor.datasets')

        # set target variable and join_key
        df_application_train.target = df_application_train.data['TARGET']
        df_application_train.join_key = ['SK_ID_CURR']

        # set exclude vars
        df_application_train.exclude_vars = ['TARGET', 'SK_ID_CURR']

        # ================
        # num_vars and cat_vars
        # ================

        # set some num_vars
        df_application_train.num_vars = ['CNT_CHILDREN',\
        'AMT_INCOME_TOTAL',\
        'AMT_CREDIT',\
        'AMT_ANNUITY',\
        'AMT_GOODS_PRICE',\
        'OWN_CAR_AGE']

        # set some cat_vars
        df_application_train.cat_vars = ['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']

        # this is super dirty right now
        current_cols = ['REGION_POPULATION_RELATIVE',\
        'EXT_SOURCE_1',\
        'EXT_SOURCE_2',\
        'EXT_SOURCE_3',\
        'APARTMENTS_AVG',\
        'BASEMENTAREA_AVG',\
        'YEARS_BEGINEXPLUATATION_AVG',\
        'YEARS_BUILD_AVG',\
        'COMMONAREA_AVG',\
        'ELEVATORS_AVG',\
        'ENTRANCES_AVG',\
        'FLOORSMAX_AVG',\
        'FLOORSMIN_AVG',\
        'LANDAREA_AVG',\
        'LIVINGAPARTMENTS_AVG',\
        'LIVINGAREA_AVG',\
        'NONLIVINGAPARTMENTS_AVG',\
        'NONLIVINGAREA_AVG',\
        'APARTMENTS_MODE',\
        'BASEMENTAREA_MODE',\
        'YEARS_BEGINEXPLUATATION_MODE',\
        'YEARS_BUILD_MODE',\
        'COMMONAREA_MODE',\
        'ELEVATORS_MODE',\
        'ENTRANCES_MODE',\
        'FLOORSMAX_MODE',\
        'FLOORSMIN_MODE',\
        'LANDAREA_MODE',\
        'LIVINGAPARTMENTS_MODE',\
        'LIVINGAREA_MODE',\
        'NONLIVINGAPARTMENTS_MODE',\
        'NONLIVINGAREA_MODE',\
        'APARTMENTS_MEDI',\
        'BASEMENTAREA_MEDI',\
        'YEARS_BEGINEXPLUATATION_MEDI',\
        'YEARS_BUILD_MEDI',\
        'COMMONAREA_MEDI',\
        'ELEVATORS_MEDI',\
        'ENTRANCES_MEDI',\
        'FLOORSMAX_MEDI',\
        'FLOORSMIN_MEDI',\
        'LANDAREA_MEDI',\
        'LIVINGAPARTMENTS_MEDI',\
        'LIVINGAREA_MEDI',\
        'NONLIVINGAPARTMENTS_MEDI',\
        'NONLIVINGAREA_MEDI',\
        'FONDKAPREMONT_MODE',\
        'HOUSETYPE_MODE',\
        'TOTALAREA_MODE',\
        'WALLSMATERIAL_MODE',\
        'EMERGENCYSTATE_MODE']

        for col in current_cols:

        if type(df_application_train.data[col][0]) == str:
            
            # set more cat_vars
            cat_vars.append(col)

        else:

            # set more num_vars
            num_vars.append(col)

        df_application_train.data['PERC_FAM_MEMBERS_CHILDREN'] = (1.0 * df_application_train.data.CNT_CHILDREN) / df_application_train.data.CNT_FAM_MEMBERS

        df_application_train.num_vars += ['CNT_FAM_MEMBERS','PERC_FAM_MEMBERS_CHILDREN']
        df_application_train.num_vars += ['REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']
        df_application_train.num_vars += ['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']
        df_application_train.num_vars += ['AMT_REQ_CREDIT_BUREAU_HOUR',\
                    'AMT_REQ_CREDIT_BUREAU_DAY',\
                    'AMT_REQ_CREDIT_BUREAU_WEEK',\
                    'AMT_REQ_CREDIT_BUREAU_MON',\
                    'AMT_REQ_CREDIT_BUREAU_QRT',\
                    'AMT_REQ_CREDIT_BUREAU_YEAR']

        cat_vars += ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START']

        df_application_train.data.WEEKDAY_APPR_PROCESS_START = df_application_train.data.WEEKDAY_APPR_PROCESS_START.apply(lambda x: str(x))
        df_application_train.data.HOUR_APPR_PROCESS_START = df_application_train.data.HOUR_APPR_PROCESS_START.apply(lambda x: str(x))

        df_application_train.data['ORGANIZATION_TYPE_GROUP'] = df_application_train.data.ORGANIZATION_TYPE.apply(lambda x: x.split(':')[0].split('Type')[0])
        df_application_train.cat_vars += ['ORGANIZATION_TYPE','ORGANIZATION_TYPE_GROUP']

        df_application_train.cat_features = df_application_train.data[df_application_train.cat_vars]
        df_application_train.num_features = df_application_train.data[df_application_train.num_vars]

        # ================
        # binary vars
        # ================

        df_application_train.discover_binary_vars()
        df_application_train.transform_binary_features()

        # ================
        # concatenating everything
        # ================

        all_df_application_train_features_dfs = [df_application_train.cat_features,df_application_train.num_features,\
        df_application_train.binary_features]

        if include_join_key == True:

            all_df_application_train_features_dfs.append(df_application_train.data[df_application_train.join_key])

        self.df_application_train_features = pd.concat(all_df_application_train_features_dfs, axis=1)

        # we can get dummies after concatenating all of the processed datasets
        # self.df_application_train_features = pd.get_dummies(self.df_application_train_features)