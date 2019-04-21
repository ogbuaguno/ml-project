from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import un_transforms as tr

class TransformFeatureColumns:
    
    #set current timestamp and use as endpoint. transform business origin date and date business was created to ages
    current_timestamp = int(pd.Timestamp.timestamp(pd.Timestamp('2019-01-31')))
    
    def __init__(self, X):
        self.X = X
        self.oheC = OneHotEncoder(handle_unknown='ignore')
        self.oheB = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
    
    def splitData(self, split):
        X_split = self.X.reset_index()
        X_train, X_test = tr.split_train_test_by_id(X_split, split, "index")
        self.X_train = X_train
        self.X_test = X_test

    def spoolData(self, X, daily_trans_instrument_count):
        #initialize dataframe
        unsupervised = pd.DataFrame()
        
        #the business_age is the difference between the current timestamp (set as December 31st 2018) and when the business got 
        #it's first successful transaction
        business_age = tr.processBusinessDates(X['origin_date'].values, self.current_timestamp)
        unsupervised['business_age'] = business_age
                   
        #the business_first_tranx_age is the difference between when the business was created and when it had it's first 
        #successful transaction
        business_first_trnx_age = tr.processAges(X['origin_date'].values, X['created_at'].values)
        unsupervised['business_first_trnx_age'] = business_first_trnx_age
        
        is_registered, is_starter = tr.processBusinessType(X['business_type'].values)
        unsupervised['is_registered'] = is_registered
        unsupervised['is_starter'] = is_starter
        
        #the customer_age is the difference between the transaction date and when the customer was created
        customer_age = tr.processAges(X['transaction_date'].values, X['customer_created_on'].values)
        unsupervised['customer_age'] = customer_age
        
        #the customer_first_trnx_age is the difference between the customer's first successful transaction and when
        #the customer was created
        customer_first_trnx_age = tr.processAges(X['customer_first_transaction'].values, X['customer_created_on'].values)
        unsupervised['customer_first_trnx_age'] = customer_first_trnx_age
        
        #the daily_trans_instrument_count uses the instrument encoding and transaction date to count how many times an instrument
        #has been used for transactions in one day
        daily_fingerprint_count, daily_trans_instrument_count = tr.getDailyTransInstrumentCount(X['transaction_date'].values, 
        X['fingerprint'].values, daily_trans_instrument_count)
        unsupervised['daily_trans_instrument_count'] = daily_trans_instrument_count
        
        #the is_local_instrument checks if the instrument country is Nigeria
        is_local_instrument = tr.checkIsLocalInstrument(X['country_name'].values)
        unsupervised['is_local_instrument'] = is_local_instrument
        
        #transform the payment channel to 3 columns using binarization
        paid_with_card, paid_with_bank, paid_with_other = tr.processPaymentChannels(X['channel'].values)

        unsupervised['paid_with_card'] = paid_with_card
        unsupervised['paid_with_bank'] = paid_with_bank
        unsupervised['paid_with_other'] = paid_with_other

        #unsupervised['amount'] = X['amount'].copy().apply(lambda n: n/100)
        unsupervised['amount'] = X['amount'].values/100 #np.true_divide(X['amount'].values, 1000)
        
        Y = tr.encodeLabel(X['category'].values)
        
        return daily_fingerprint_count, unsupervised, Y
        
    #transforms the training data
    def spoolTrainingData(self):
        X = self.X_train
        
        daily_fingerprint_count, unsupervised, Y = self.spoolData(X, {})
        self.daily_fingerprint_count = daily_fingerprint_count
        
        #unsupervised.describe()
        
        #transforms the currency using one hot encoding
        currencies = self.oheC.fit_transform(np.array(X['currency']).reshape(-1,1))
        
        for i, col in enumerate(self.oheC.get_feature_names()) :
            unsupervised[col] = pd.SparseSeries(currencies[:,i].toarray().ravel(), fill_value=0)
            
        #transforms the banks using one hot encoding
        bankEncoded = self.oheB.fit_transform(np.array(X['bank']).reshape(-1,1))

        for i, col in enumerate(self.oheB.get_feature_names()) :
            unsupervised[col] = pd.SparseSeries(bankEncoded[:,i].toarray().ravel(), fill_value=0)
        
        return unsupervised, Y
    
    def spoolOrangeTrainingData(self):
        X = self.X_train
        
        daily_fingerprint_count, unsupervised, Y = self.spoolData(X, {})
        self.daily_fingerprint_count = daily_fingerprint_count
        
        unsupervised['currency'] = X['currency'].values #.apply(lambda x: 'NGN' if (x == '?') else x)
        unsupervised['bank'] = X['bank'].values #.apply(lambda x: 'NGN' if (x == '?') else x)
        unsupervised['label'] = X['category'].values
        
        return unsupervised
        
    #transforms the test data
    def spoolTestData(self, testData=None):
        if not testData is None:
            X = testData
        else:
            X = self.X_test
        
        daily_fingerprint_count, unsupervised, Y = self.spoolData(X, self.daily_fingerprint_count)
        
        currencies = self.oheC.transform(np.array(X['currency']).reshape(-1,1))
        for i, col in enumerate(self.oheC.get_feature_names()) :
            unsupervised[col] = pd.SparseSeries(currencies[:,i].toarray().ravel(), fill_value=0)
        
        bankEncoded = self.oheB.transform(np.array(X['bank']).reshape(-1,1))
        for i, col in enumerate(self.oheB.get_feature_names()) :
            unsupervised[col] = pd.SparseSeries(bankEncoded[:,i].toarray().ravel(), fill_value=0)
        
        return unsupervised, Y
    
    def spoolOrangeTestData(self, testData=None):
        if not testData is None:
            X = testData
        else:
            X = self.X_test
        
        daily_fingerprint_count, unsupervised, Y = self.spoolData(X, self.daily_fingerprint_count)
        
        unsupervised['currency'] = X['currency'].values#.apply(lambda x: x if not (x == '?') else 'NGN')
        unsupervised['bank'] = X['bank'].values#.apply(lambda x: x if not (x == '?') else 'Unknown')
        unsupervised['label'] = X['category'].values
        
        return unsupervised
    
    #scales the data according to training data
    def scaleData(self, X, trainingData=False) :
        if trainingData:
            self.scaler.fit(X.values)
            
        X_np = self.scaler.transform(X.values)
        XScaled = pd.DataFrame(X_np, index=X.index, columns=X.columns)
                                  
        return XScaled
        