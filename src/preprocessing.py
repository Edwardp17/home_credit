# Import libraries
import os
import gc
gc.enable()
import random
import numpy as np 
import pandas as pd
from scipy import stats

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
        if self.data == None:

            raise Exception('Data cannot be None for binary vars to be discovered.')

        if self.binary_vars == None:

                raise Warning('Tried to transform binary vars but found no binary vars. Making no transformations.')

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


    def set_application_train_features(self, application_train_name = 'application_train'):

        df_application_train = None
        for df in self.datasets:

            if df.name == application_train_name:

                df_application_train = df
        
        if df_application_train = None:

            raise Exception('Could not find application_train in Preprocessor.datasets')

        
        # ================
        # Target variable, join key, exclude vars
        # ================

        df_application_train.target = df_application_train.data['TARGET']
        df_application_train.join_key = ['SK_ID_CURR']
        df_application_train.exclude_vars = ['TARGET']

        # ================
        # Cleaning
        # ================

        df_application_train.data.WEEKDAY_APPR_PROCESS_START = df_application_train.data.WEEKDAY_APPR_PROCESS_START.apply(lambda x: str(x))
        df_application_train.data.HOUR_APPR_PROCESS_START = df_application_train.data.HOUR_APPR_PROCESS_START.apply(lambda x: str(x))

        # ================
        # Feature engineering
        # ================

        # returns a modified dataframe that has num var percentiles partitioned by each cat var
        def num_percentiles_partitioned_by_cat_var(df):

            num_vars = []
            cat_vars = []
            for col in df.columns:

                if type(df[col][0]) in (int, float):

                    num_vars.append(col)
                
                elif type(df[col][0]) in (str):

                    cat_vars.append(col)

            for cat_var in cat_vars:
                for num_var in num_vars:

                    df[str(cat_var) + '_' + str(num_var) + '_rankpartition'] = df.groupby(cat_var)[num_var].rank()
            
            return df

        df_application_train.data['ORGANIZATION_TYPE_GROUP'] = df_application_train.data.ORGANIZATION_TYPE.apply(lambda x: x.split(':')[0].split('Type')[0])
        df_application_train.data['PERC_FAM_MEMBERS_CHILDREN'] = (1.0 * df_application_train.data.CNT_CHILDREN) / df_application_train.data.CNT_FAM_MEMBERS
        
        # get num var percentiles partitioned by each cat var
        df_application_train.data = num_percentiles_partitioned_by_cat_var(df_application_train.data)

        # ================
        # Standardize binary vars
        # ================

        df_application_train.discover_binary_vars()
        df_application_train.transform_binary_features()

        # ================
        # Get final feature set
        # ================

        self.df_application_train_features = df_application_train.data[[x for x in df_application_train.data.columns if x not in df_application_train.exclude_vars]]

        # we can get dummies after concatenating all of the processed datasets
        # self.df_application_train_features = pd.get_dummies(self.df_application_train_features)