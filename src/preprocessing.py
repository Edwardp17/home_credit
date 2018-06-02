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

        for col in self.binary_vars:

            s_binary_feature = self.data[col]
            unique_values = list(s_binary_feature.unique())

            if set(unique_values) != set([0,1]):
                
                standardization_map = dict(zip(unique_values),[0,1])
                s_binary_feature = [standardization_map[x] for x in list(s_binary_feature)]
                self.data[col] = s_binary_feature
        

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


    def get_application_train_features(self, application_train_name = 'application_train'):

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
        df_application_train.exclude_vars = ['TARGET']

        df_application_train.data['PERC_FAM_MEMBERS_CHILDREN'] = (1.0 * df_application_train.data.CNT_CHILDREN) / df_application_train.data.CNT_FAM_MEMBERS

        df_application_train.data.WEEKDAY_APPR_PROCESS_START = df_application_train.data.WEEKDAY_APPR_PROCESS_START.apply(lambda x: str(x))
        df_application_train.data.HOUR_APPR_PROCESS_START = df_application_train.data.HOUR_APPR_PROCESS_START.apply(lambda x: str(x))

        df_application_train.data['ORGANIZATION_TYPE_GROUP'] = df_application_train.data.ORGANIZATION_TYPE.apply(lambda x: x.split(':')[0].split('Type')[0])

        # ================
        # binary vars
        # ================

        df_application_train.discover_binary_vars()
        df_application_train.transform_binary_features()

        # ================
        # concatenating everything
        # ================

        self.df_application_train_features = df_application_train.data[[x for x in df_application_train.data.columns if x not in df_application_train.exclude_vars]]

        # we can get dummies after concatenating all of the processed datasets
        # self.df_application_train_features = pd.get_dummies(self.df_application_train_features)