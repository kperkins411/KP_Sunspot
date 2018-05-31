from unittest import TestCase, main
from lag_queue import Lag_q,track_preds
import pandas as pd
import sys
import copy as copy
'''
This file intended to test classes in lag_queue.py
pycharm project name SunSpot
'''
sys.path.insert(0, '/home/keith/fastAI2/fastai/courses/ml1/')
num_lags = 12

class dummymodel(object):
    def __init__(self):
        super().__init__()
        self.pred = 0

    def predict_dl(self, dl):
        self.pred +=1
        return [[self.pred]]  #same as output of real predict

def delete_lag_nas_in_mapper(mapper):
    '''
    get rid of the lag_na columns in mapper
    :param mapper:
    :return:
    '''
    for i in range(len(mapper.features) - 1, 0, -1):
        if 'na' in mapper.features[i][0][0]:
            del (mapper.features[i])
    for i in range(len(mapper.built_features) - 1, 0, -1):
        if 'na' in mapper.built_features[i][0][0]:
            del (mapper.built_features[i])
    for i in range(len(mapper.transformed_names_) - 1, 0, -1):
        if 'na' in mapper.transformed_names_[i]:
            del (mapper.transformed_names_[i])

def dump_na_columns_in_dataframe(df):
    '''
    get rid of the lag_n_na fields in a dataframe
    :param df1:
    :return:
    '''

    # for _, row in df.iterrows():
    for i in range(1,num_lags+1):
        try:
            df.drop(f'lag_{i}_na',axis=1,inplace=True)
        except KeyError as e:
            print('KeyError:{0}'.format(e))

class Lag_qTestCase(TestCase):
    def setUp(self):
        pass
    def test_add_pred(self):
        with self.assertRaises(ValueError):
            Lag_q(0)

        nLags = 2
        a = self.create(nLags)

        self.assertEquals(a.lag_q[0], 3)
        self.assertEquals(a.lag_q[1], 2)
        self.assertEquals(len(a.lag_q), 2)

    def test_update_lag(self):
        a = self.create(2)

        #confirm dataframe fail
        d = {"lag_1": [1], "lag_2": [2]}
        df = pd.DataFrame(data=d)
        with self.assertRaises(AssertionError):
            a.update_lag(df)

        #confirm series pass
        d={"lag_1":[7],"lag_2":[8]}
        df = pd.DataFrame(data = d)
        ds = df.iloc[0]
        ds = a.update_lag(ds)
        self.assertEqual(ds.lag_1, 3)
        self.assertEqual(ds.lag_2, 2)

    def create(self,max_lags):
        a = Lag_q(max_lags)
        a.add_pred(1)
        a.add_pred(2)
        a.add_pred(3)
        return a

class track_predsTestCase(TestCase):
    def setUp(self):
        super().setUp()

        self.num_lags =12
        PATH = "./SS_Data/"

        # get mapper
        import pickle
        input = open(f'{PATH}/mapper', 'rb')
        self.mapper = pickle.load(input)
        input.close()
        delete_lag_nas_in_mapper(self.mapper)

        #get mapper with mean of 0 and standard deviation of 1
        self.mapper_unit = copy.deepcopy(self.mapper)
        for lag in range(1, self.num_lags + 1):
            self.mapper_unit.features[lag - 1][1].mean_ = 0.0
            self.mapper_unit.features[lag - 1][1].var_ = 1.0

        # get dataframe
        import pandas as pd
        PATH = "./SS_Data/"
        self.df = pd.read_feather(f'{PATH}df')
        self.df = self.df.iloc[0:20]        # get a slice

        # get rid of na columns
        dump_na_columns_in_dataframe(self.df)

        #zero out all the lag columns
        for i in range(self.num_lags):
            self.df[f'lag_{i+1}'] = 0

        self.cat_vars = ["Year", "Month", "Dayofyear", "Is_quarter_end", "Is_quarter_start", "Is_year_end", "Is_year_start"]

    def test_run(self):
        #first test with unit mapper
        md = dummymodel()
        self.p = track_preds(df=self.df, mapper=self.mapper_unit, model=md, cat_vars=self.cat_vars, num_lags=self.num_lags)
        self.p.run()

        # for index in range(len(self.df)):
        column_sum = 190.0 # what first column should be
        subtractor = 19
        for lag in range(1, self.num_lags + 1):
            self.assertAlmostEqual(self.p.df[[f'lag_{lag}']].sum()[0], column_sum, 1)
            column_sum-=subtractor
            subtractor-=1

        #now with real mapper
        md = dummymodel()
        self.p = track_preds(df=self.df, mapper=self.mapper, model=md,cat_vars=self.cat_vars, num_lags=self.num_lags)
        self.p.run()

        pass

    def tearDown(self):
        super().tearDown()


if __name__ == '__main__':
    main()

