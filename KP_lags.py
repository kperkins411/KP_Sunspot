from collections import deque
from fastai.structured import *
from fastai.column_data import *
import timeit
'''
these functions expect a df no 'na' columns
'''

def delete_lag_nas_in_mapper(mapper):
    '''
    get rid of the lag_na columns in mapper
    :param mapper:
    :return:
    '''
    for i in range(len(mapper.features)-1,0,-1):
        if 'na' in mapper.features[i][0][0]:
            del (mapper.features[i])
    for i in range(len(mapper.built_features)-1,0,-1):
        if 'na' in mapper.built_features[i][0][0]:
            del (mapper.built_features[i])
    for i in range(len(mapper.transformed_names_) - 1, 0, -1):
        if 'na' in mapper.transformed_names_[i]:
            del (mapper.transformed_names_[i])

class Lag_q(object):
    '''
    maintains a list of unscaled lags

    typically call predict
    then add_pred
    then update_lag
    '''

    def __init__(self, max_lags=12):
        self.max_lags = max_lags
        self.lag_q = deque(maxlen=self.max_lags)

    def add_pred(self, prediction):
        # add prediction for row-n
        self.lag_q.appendleft(prediction)

    def update_lag(self, df):
        # move values in lag_q to df.lag_n values
        # expects df to be a single row

        # #make sure its a series
        # if (len(df) > 1):
        #     raise  ValueError("df is %s"%str(len(df)))
#         print(df.loc[:,'lag_1'])
#         print(self.lag_q[0][0])
        
        try:
            for i in range(len(self.lag_q)):
                df.loc[f'lag_{i+1}'] = self.lag_q[i][0]
        except:
            print("problem in lag_q:update_lag")
            raise
        return df


class track_preds(object):
    '''
    pass in a dataframe, this class will loop through
    call model.predict on each row
    and then update the lags for the next row accordingly

    the dataframe lags should not be scaled initially
    '''

    def __init__(self, df,mapper, model,cat_vars, num_lags):
        '''

        :param df: single or multiple row dataframe
        :param model: model that runs predictions
        :param cat_vars: list of categorical variables in df
        '''
        self.df = df
        self.mapper = mapper
        self.model = model
        self.cat_vars = cat_vars
        self.lag_q = Lag_q(num_lags)
        self.preds = []
        self.num_lags = num_lags
    
    def run(self):
#         delete_lag_nas_in_mapper(self.mapper)
#         self.mapper.df_out = True

        for index in range(len(self.df)-2):             

            # update the lags with the latest predictions
            self.df.iloc[index] = self.lag_q.update_lag(self.df.iloc[index])
 
#             self.lag_q.update_lag(self.df.iloc[index])
            
            # following 2 lines work but are slow
            # one_row = self.df[index:index + 1]
            # one_row[self.mapper.transformed_names_] = self.mapper.transform(one_row)

            # subtract the mean and divide by std dev for each row
            for lag in range(1, self.num_lags + 1):
                # lags start at 1 but are 0 indexed, so sub 1 from lag number
                # the 1 is the location of the StandardScaler in the tuple
                self.df.loc[index,f'lag_{lag}'] = ((self.df.loc[index,f'lag_{lag}'] - self.mapper.features[lag - 1][1].mean_) / np.sqrt(self.mapper.features[lag - 1][1].var_))

            # get next row, create dataloader for it and then run prediction on it
            q=self.df.iloc[index:(index + 2),:]
            one_row = ColumnarDataset.from_data_frame(q, cat_flds=self.cat_vars)
 
            # get a data loader
            dl1 = DataLoader(one_row)

            # prediction = np.exp(self.model.predict_dl(dl1))
            prediction = self.model.predict_dl(dl1)
   
            # add to predictions
            self.preds.append(prediction[0][0])

            # add that prediction to lags
            self.lag_q.add_pred(np.exp(prediction))

######################################################
class dummymodel(object):
    pred = 0
    def predict_dl(self, dl):
        dummymodel.pred +=1
        return dummymodel.pred
# class track_preds1(object):
#     '''
#     pass in a dataframe, this class will loop through
#     call model.predict on each row
#     and then update the lags for the next row accordingly

#     the dataframe lags should not be scaled initially
#     '''

#     def __init__(self, df,mapper, model,cat_vars, num_lags):
#         '''

#         :param df: single or multiple row dataframe
#         :param model: model that runs predictions
#         :param cat_vars: list of categorical variables in df
#         '''
#         self.df = df
#         self.mapper = mapper
#         self.model = model
#         self.cat_vars = cat_vars
#         self.lag_q = Lag_q(num_lags)
#         self.preds = []
#         self.num_lags = num_lags
    
#     def run(self):
#         for index in range(len(self.df)-2):
            
#             one_row = ColumnarDataset.from_data_frame(self.df.iloc[index:(index+2),:],cat_flds=self.cat_vars)
#             one_row_test = DataLoader(one_row)
#             prediction = self.model.predict_dl(one_row_test)
#             self.preds.append(prediction[0][0])
    
# class m_lag(object):
#     def __init__(self, df,mapper, model,cat_vars, num_lags):
#         self.df = df
#         self.mapper = mapper
#         self.model = model
#         self.cat_vars = cat_vars
# #         self.lag_q = Lag_q(num_lags)
#         self.preds = []
#         self.num_lags = num_lags
#     def run1(self):
#         for index in range(len(self.df)-2):
#             one_row = ColumnarDataset.from_data_frame(self.df.iloc[index:(index+2),:],cat_flds=self.cat_vars)
#             one_row_test = DataLoader(one_row)
#             prediction = self.model.predict_dl(one_row_test)
#             self.preds.append(prediction[0][0])
    
