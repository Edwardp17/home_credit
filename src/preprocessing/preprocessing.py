# Import libraries
import os
import gc
gc.enable()
import random
import numpy as np 
import pandas as pd
import featuretools as ft
from scipy import stats

DATA_PATH = '../data/'
DATASET_FILENAMES = []


class Dataset:


    def __init__(self, filename = None, join_key = None, exclude_vars = [],\
     cat_vars = [], num_vars = [], binary_vars = [], data = None):
        
        if filename != None:
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

        self.data = data
        self.features = None
        self.target = None


    def read_data(self):

        self.data = pd.read_csv(DATA_PATH + self.filename)


    def discover_binary_vars(self):
        
        if self.data == None:

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

    # returns a modified dataframe that has num var percentiles partitioned by each cat var
    def get_num_percentiles_partitioned_by_cat_var(self, cat_vars = [], num_vars = []):

        if cat_vars == []:
            cat_vars = [col for col in self.data.columns if self.data[col].dtype in [str,'O'] and col != 'TARGET']

        if num_vars == []:
            num_vars = [col for col in self.data.columns if self.data[col].dtype in [int, float] and col != 'TARGET']
        print(len(cat_vars))
        print(len(num_vars))
        c = []
        for i, cat_var in enumerate(cat_vars):
            print('Round ' + str(i+1))
            print(str(pd.Timestamp.now()))
            for num_var in num_vars:
                self.data[str(cat_var) + '_' + str(num_var) + '_rankpartition'] = self.data.groupby(cat_var)[num_var].rank()
            c.append(cat_var)
            print('Percentage done: ' + str(len(c)/len(cat_vars)))
        

class Preprocessor:


    def __init__(self):
        
        self.datasets = []

        self.df_application_train_features = None
        self.df_bureau_features = None

        self.featuretools_feature_set = None
        self.featuretools_feature_names = None

    # ================
    # Class-level helper functions
    # ================

    def read_all_data(self, dataset_filenames=[]):

        datasets = []

        for d in dataset_filenames:

            dataset = Dataset(filename = d)
            dataset.read_data()
            datasets.append(dataset)

        self.datasets = datasets


    # ================================
    # Featuretools
    # ================================

    def run_featuretools(self, read_in_data_if_needed = True, export_to_csv = False):

        # TODO: This should eventually be dynamic.
        dataset_filenames = ['POS_CASH_balance.csv', 'application_test.csv', 'application_train.csv', 'bureau.csv',\
        'bureau_balance.csv', 'credit_card_balance.csv', 'installments_payments.csv', 'previous_application.csv']

        if self.datasets == []:
            self.read_all_data(dataset_filenames = dataset_filenames)
        for data in self.datasets:
            if data.name == 'POS_CASH_balance':
                pos = data.data
            elif data.name == 'application_test':
                test = data.data
            elif data.name == 'application_train':
                train_full = data.data
            elif data.name == 'bureau':
                bureau = data.data
            elif data.name == 'bureau_balance':
                bureau_balance = data.data
            elif data.name == 'credit_card_balance':
                cc_bal = data.data
            elif data.name == 'installments_payments':
                inst = data.data
            elif data.name == 'previous_application':
                prev_app = data.data
        
        train = train_full.drop('TARGET',axis = 1)
        train_y = train_full['TARGET']

        print('Creating entity set.')

        # Create new entityset
        es = ft.EntitySet(id = 'train')
        print('Creating train entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'train', dataframe = train, index = 'SK_ID_CURR')
        print('Creating bureau entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')
        print('Creating bureau_bal entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'bureau_bal', dataframe = bureau_balance, make_index = True, index = 'bureau_bal_id')
        print('Creating pos entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'pos', dataframe = pos, make_index = True, index = 'pos_id')
        print('Creating cc_bal entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'cc_bal', dataframe = cc_bal, make_index = True, index = 'cc_bal_id')
        print('Creating inst entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'inst', dataframe = inst, make_index = True, index = 'inst_id')
        print('Creating prev_app entity.')
        print(str(pd.Timestamp.now()))
        es = es.entity_from_dataframe(entity_id = 'prev_app', dataframe = prev_app, index = 'SK_ID_PREV')

        print('Creating relationships.')
        print(str(pd.Timestamp.now()))

        # Create relationships
        print('Creating r_train_bureau.')
        print(str(pd.Timestamp.now()))
        r_train_bureau = ft.Relationship(es['train']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
        es = es.add_relationship(r_train_bureau)
        
        print('Creating r_bureau_bureau_bal.')
        print(str(pd.Timestamp.now()))
        r_bureau_bureau_bal = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_bal']['SK_ID_BUREAU'])
        es = es.add_relationship(r_bureau_bureau_bal)
        
        print('Creating r_train_pos.')
        print(str(pd.Timestamp.now()))
        r_train_pos = ft.Relationship(es['train']['SK_ID_CURR'], es['pos']['SK_ID_CURR'])
        es = es.add_relationship(r_train_pos)
        
        print('Creating r_train_cc_bal.')
        print(str(pd.Timestamp.now()))
        r_train_cc_bal = ft.Relationship(es['train']['SK_ID_CURR'], es['cc_bal']['SK_ID_CURR'])
        es = es.add_relationship(r_train_cc_bal)

        print('Creating r_train_inst.')
        print(str(pd.Timestamp.now()))
        r_train_inst = ft.Relationship(es['train']['SK_ID_CURR'], es['inst']['SK_ID_CURR'])
        es = es.add_relationship(r_train_inst)

        print('Creating r_train_prev_app.')
        print(str(pd.Timestamp.now()))
        r_train_prev_app = ft.Relationship(es['train']['SK_ID_CURR'], es['prev_app']['SK_ID_CURR'])
        es = es.add_relationship(r_train_prev_app)

        print('Creating r_prev_app_pos.')
        print(str(pd.Timestamp.now()))
        r_prev_app_pos = ft.Relationship(es['prev_app']['SK_ID_PREV'], es['pos']['SK_ID_PREV'])
        es = es.add_relationship(r_prev_app_pos)

        print('Creating r_prev_app_inst.')
        print(str(pd.Timestamp.now()))
        r_prev_app_inst = ft.Relationship(es['prev_app']['SK_ID_PREV'], es['inst']['SK_ID_PREV'])
        es = es.add_relationship(r_prev_app_inst)

        print('Creating r_prev_app_cc_bal.')
        print(str(pd.Timestamp.now()))
        r_prev_app_cc_bal = ft.Relationship(es['prev_app']['SK_ID_PREV'], es['cc_bal']['SK_ID_PREV'])
        es = es.add_relationship(r_prev_app_cc_bal)
        
        # Create new features using specified primitives
        # Documentation: https://docs.featuretools.com/generated/featuretools.dfs.html

        print('Creating actual features.')
        print(str(pd.Timestamp.now()))
        features, feature_names = ft.dfs(entityset = es, target_entity = 'train', \
        agg_primitives = ['mean', 'max', 'last'], \
        trans_primitives = ['years', 'month', 'subtract', 'divide'])

        self.featuretools_feature_set = features
        self.featuretools_feature_names = feature_names

        print('Done running featuretools!')

        print('Exporting features to CSV.')

        if export_to_csv:
            pd.DataFrame(features).to_csv('featuretools_feature.csv')
    
    # ================================
    # Manual FE
    # ================================

    # ================
    # application_train
    # ================

    def set_application_train_features(self, dataset_name = 'application_train'):

        df_application_train = None
        for df in self.datasets:

            if df.name == dataset_name:

                df_application_train = df
        
        if df_application_train == None:

            raise Exception('Could not find application_train in Preprocessor.datasets')
        
        # ================
        # Target variable, join key, exclude vars
        # ================
        if dataset_name == 'application_train':
            df_application_train.target = df_application_train.data['TARGET']
            df_application_train.exclude_vars = ['TARGET']
        df_application_train.join_key = ['SK_ID_CURR']
        # ================
        # Cleaning
        # ================

        df_application_train.data.WEEKDAY_APPR_PROCESS_START = df_application_train.data.WEEKDAY_APPR_PROCESS_START.apply(lambda x: str(x))
        df_application_train.data.HOUR_APPR_PROCESS_START = df_application_train.data.HOUR_APPR_PROCESS_START.apply(lambda x: str(x))

        # ================
        # Feature engineering
        # ================

        df_application_train.data['ORGANIZATION_TYPE_GROUP'] = df_application_train.data.ORGANIZATION_TYPE.apply(lambda x: x.split(':')[0].split('Type')[0])
        df_application_train.data['PERC_FAM_MEMBERS_CHILDREN'] = (1.0 * df_application_train.data.CNT_CHILDREN) / df_application_train.data.CNT_FAM_MEMBERS
        
        df_application_train.get_num_percentiles_partitioned_by_cat_var()

        # ================
        # Standardize binary vars
        # ================

        # df_application_train.discover_binary_vars()
        # df_application_train.transform_binary_features()

        # ================
        # Get final feature set
        # ================

        self.df_application_train_features = df_application_train.data[[x for x in df_application_train.data.columns if x not in df_application_train.exclude_vars]]

        print('df_application_train preprocessed. df.data.shape:' + str(df.data.shape))
        # we can get dummies after concatenating all of the processed datasets
        # self.df_application_train_features = pd.get_dummies(self.df_application_train_features)


    def set_bureau_train_features(self, dataset_name = 'bureau'):

        df_bureau = None
        for df in self.datasets:

            if df.name == dataset_name:

                df_bureau = df
        
        if df_bureau == None:

            raise Exception('Could not find application_train in Preprocessor.datasets')

        # ================
        # Target variable, join key, exclude vars
        # ================

        df_bureau.join_key = 'SK_ID_CURR'

        # ================
        # Cleaning
        # ================

        # ================
        # Feature engineering
        # ================

        # get counts of previous loans
        df_bureau_formatted = pd.DataFrame(df_bureau.data.groupby(by='SK_ID_CURR')['SK_ID_CURR'].count()).rename(columns={'SK_ID_CURR':'SK_ID_CURR_count'})

        # get counts of former credit, by status
        # TODO: these functions by need an added 'self'
        def get_counts_by( df_bureau, feature_agg_col):
            df_agg = pd.DataFrame(df_bureau.data.groupby(by=[df_bureau.join_key,feature_agg_col])[feature_agg_col].count()).rename(columns={feature_agg_col:feature_agg_col + '_count'}).reset_index()
            df_agg.index = df_agg.SK_ID_CURR

            df_feature_count_full = pd.DataFrame()
            for unique_val in df_bureau.data[feature_agg_col].unique():
                df_feature_count = pd.DataFrame(df_agg.loc[df_agg[feature_agg_col] == unique_val, feature_agg_col + '_count'])
                df_feature_count = df_feature_count.rename(columns={feature_agg_col + '_count':feature_agg_col + '_count_' + unique_val})
                if len(df_feature_count_full) > 0:
                    df_feature_count_full = df_feature_count_full.merge(df_feature_count,how='left',left_index=True,right_index=True)
                else:
                    df_feature_count_full = df_feature_count[:]

            return df_feature_count_full
        
        df_status_count_full = get_counts_by(df_bureau, feature_agg_col = 'CREDIT_ACTIVE')

        df_bureau_formatted.merge(df_status_count_full,how='left',left_index=True,right_index=True)

        # get unique number of currencies that previous credits exist in, for each current application
        df_bureau_formatted.merge(pd.DataFrame(df_bureau.data.groupby(by='SK_ID_CURR')['CREDIT_CURRENCY'].nunique()).rename(columns={'CREDIT_CURRENCY':'CREDIT_CURRENCY_unique'}),\
        how='left',left_index=True,right_index=True)

        # get most recent number of days a credit was extend before the current application
        df_bureau_formatted.merge(pd.DataFrame(df_bureau.data.groupby(by='SK_ID_CURR')['DAYS_CREDIT'].min()).rename(columns={'DAYS_CREDIT':'DAYS_CREDIT_min'}),\
        how='left',left_index=True,right_index=True)

        # TODO: get average number of days in between applications

        # TODO: these functions by need an added 'self'
        def get_min_average_max( df_bureau_formatted, col, min = True, average = True, max = True):
            if min:
                # get min
                df_bureau_formatted.merge(pd.DataFrame(df_bureau.data.groupby(by='SK_ID_CURR')[col].min()).rename(columns={col:col+'_min'}),\
                how='left',left_index=True,right_index=True)

            if average:
                # get average
                df_bureau_formatted.merge(pd.DataFrame(df_bureau.data.groupby(by='SK_ID_CURR')[col].mean()).rename(columns={col:col+'_mean'}),\
                how='left',left_index=True,right_index=True)

            if max:
                # get max
                df_bureau_formatted.merge(pd.DataFrame(df_bureau.data.groupby(by='SK_ID_CURR')[col].max()).rename(columns={col:col+'_max'}),\
                how='left',left_index=True,right_index=True)

            return df_bureau_formatted

        # TODO: Not sure if DAYS_CREDIT_UPDATE belongs in here
        for col in ['CREDIT_DAY_OVERDUE','DAYS_CREDIT_ENDDATE','AMT_CREDIT_MAX_OVERDUE',\
        'CNT_CREDIT_PROLONG','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_UPDATE',\
        'AMT_ANNUITY']:
            df_bureau_formatted = get_min_average_max(df_bureau_formatted,col)
        
        df_credit_type_count_full = get_counts_by(df_bureau, feature_agg_col = 'CREDIT_TYPE')

        df_bureau_formatted.merge(df_credit_type_count_full,how='left',left_index=True,right_index=True)

        # ================
        # Standardize binary vars
        # ================

        df_bureau_formatted_dataset = Dataset(data=df_bureau_formatted)

        # df_bureau_formatted_dataset.discover_binary_vars()
        # df_bureau_formatted_dataset.transform_binary_features()

        # ================
        # Get final feature set
        # ================

        self.df_bureau_features = df_bureau_formatted_dataset.data

        print('df_bureau preprocessed. df.data.shape:' + str(df.data.shape))