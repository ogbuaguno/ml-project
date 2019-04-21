import numpy as np
import hashlib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.externals import joblib

#processes the origin date column and expresses it in number of days
def processBusinessDates(dates, current_timestamp) : 
    age = []
    for i in range(len(dates)) :
        #if isinstance(dates[i], pd.Timestamp):
        w_date = pd.Timestamp(dates[i].astype('M8[D]'))
        w_date = int(pd.Timestamp.timestamp(w_date))
        current_age = current_timestamp - w_date
        age = np.append(age, current_age/(24*3600))
    return age

#processes ages based on a source age and destination age
def processAges(destination_dates, source_dates) :
    age = []
    for i in range(len(source_dates)):
        #if isinstance(dte, pd.Timestamp):
        trans_date = pd.Timestamp(destination_dates[i])#.astype('M8[D]'))
        trans_date = int(pd.Timestamp.timestamp(trans_date))

        cust_date = pd.Timestamp(source_dates[i])#.astype('M8[D]'))
        cust_date = int(pd.Timestamp.timestamp(cust_date))

        current_age = trans_date - cust_date
        age = np.append(age, current_age/(24*3600))
    return age
                              
#processes the business type column and expresses it in 2 boolean columns of is_registered and is_starter
def processBusinessType(business_types):
    is_registered = []
    is_starter = []
    for i in range(len(business_types)) :
        if business_types[i] == 'registered':
            is_registered = np.append(is_registered, 1)
            is_starter = np.append(is_starter, 0)
        else :
            is_registered = np.append(is_registered, 0)
            is_starter = np.append(is_starter, 1)
    return is_registered, is_starter
    
    
#processes the channel column and expresses it in 3 boolean columns of paid_with_bank, paid_with_card and paid_with_other
def processPaymentChannels(channels):
    paid_with_bank = []
    paid_with_card = []
    paid_with_other = []
    for i in range(len(channels)) :
        if channels[i] == 'card':
            paid_with_card = np.append(paid_with_card, 1)
            paid_with_bank = np.append(paid_with_bank, 0)
            paid_with_other = np.append(paid_with_other, 0)
        elif channels[i] == 'bank':
            paid_with_card = np.append(paid_with_card, 0)
            paid_with_bank = np.append(paid_with_bank, 1)
            paid_with_other = np.append(paid_with_other, 0)
        else :
            paid_with_card = np.append(paid_with_card, 0)
            paid_with_bank = np.append(paid_with_bank, 0)
            paid_with_other = np.append(paid_with_other, 1)
    return paid_with_card, paid_with_bank, paid_with_other
    
'''
#processes the transaction date and fingerprint columns to determine how many times an instrument has been used in a day
def getDailyTransInstrumentCount(transaction_dates, fingerprints) :
    trans_instrument_count = {}
    daily_trans_instrument_count = []
    #prev_date = (transaction_dates[0]).date()
    for i in range(len(transaction_dates)):
        curr_date = (transaction_dates[i]).date()
        curr_count = trans_instrument_count.get(fingerprints[i]) if trans_instrument_count.get(fingerprints[i]) else 0
        if (curr_date == prev_date) :
            curr_count += 1
            trans_instrument_count[fingerprints[i]] = curr_count
            daily_trans_instrument_count = np.append(daily_trans_instrument_count, curr_count)
        else :
            trans_instrument_count[fingerprints[i]] = 1
            daily_trans_instrument_count = np.append(daily_trans_instrument_count, 1)
        prev_date = curr_date
    return prev_date, trans_instrument_count, daily_trans_instrument_count
'''

#processes the transaction date and fingerprint columns to determine how many times an instrument has been used in a day
def getDailyTransInstrumentCount(transaction_dates, fingerprints, daily_fingerprint_count) :
    daily_trans_instrument_count = []
    
    if len(daily_fingerprint_count) == 0 :
        daily_fingerprint_count = {}
    
    for i in range(len(transaction_dates)):
        curr_date = transaction_dates[i].astype('M8[D]')
        fc_map = daily_fingerprint_count.get(curr_date) if daily_fingerprint_count.get(curr_date) else {}
        
        if (len(fc_map) > 0) :
            count = fc_map.get(fingerprints[i]) if fc_map.get(fingerprints[i]) else 0
            fc_map[fingerprints[i]] = count + 1
            daily_fingerprint_count[curr_date] = fc_map
            daily_trans_instrument_count = np.append(daily_trans_instrument_count, count + 1)
        else :
            fc_map[fingerprints[i]] = 1
            #fc_map_list = np.append(fc_map_list, fc_map)
            daily_fingerprint_count[curr_date] = fc_map
            daily_trans_instrument_count = np.append(daily_trans_instrument_count, 1)
    return daily_fingerprint_count, daily_trans_instrument_count
        
#processes the country name in the transaction to determine if a local instrument was used
def checkIsLocalInstrument(country_names) :
    is_local_instrument = []
    country = "nigeria"
    for i in range(len(country_names)) :
        c_name = country_names[i].lower()
        if (c_name == country) :
            is_local_instrument = np.append(is_local_instrument, 1)
        else :
            is_local_instrument = np.append(is_local_instrument, 0)
    return is_local_instrument

#splits the data set according to the test ration : taken from textbook -handsonml
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#generates a hash to split the data ino train and test set
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

#works with the function above to split the data into training and test
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

#tags a transaction as fraudulent or not
def labelTransaction(fraudTransaction, mixedTransaction) :
    category = []
    for trans in mixedTransaction :
        isContained = any(fraudTransaction == trans)
        if (isContained) :
            category = np.append(category, "fraud")
        else :
            category = np.append(category, "legit")
    return category

#encodes the label
#why manually? becuase whatever the new data, it needs to have a reference for transformation
def encodeLabel(label) :
    labelled = []
    for l in label :
        if (l == 'fraud') :
            labelled = np.append(labelled, 1)
        else :
            labelled = np.append(labelled, 0)
    return labelled

#assigns numbers to categorical values using a map
#why manually? becuase whatever the new data, it needs to have a reference for transformation
def numericToCategorical(catColumn, textMap):
    encoded = []
    if (not bool(textMap)):
        textMap = {}
        
    index = 1
    for text in catColumn :
        if (text in textMap) : 
            encoded = np.append(encoded, textMap.get(text))
        else :
            encoded[text] = index
        index += 1
    
def displayCVScores(scores):
    print("Scores \t", scores)
    print("Mean \t", scores.mean())
    print("Standard Deviation: \t", scores.std())

def saveModel(model, path) :
    with open(path, mode="w", encoding="utf-8") as sm :
        joblib.dump(model, path)
        
def crossValScores(scores, stds):
    print("Scores \t\t\t", scores)
    print("Mean \t\t\t", np.mean(scores))
    print("Standard Deviation: \t", np.mean(stds))