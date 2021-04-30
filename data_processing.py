import numpy as np
import pandas as pd
import datetime as dt

from tqdm import tqdm

import keras
import keras.backend as K
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.layers import Input, Dense, Dropout, GRU

data_path = './data/yoochoose-data/'
data_processed_path = './data/data_processed/'

data_original = pd.read_csv(data_path + 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64}, names = ['SessionId', 'TimeStr', 'ItemId'])
data_original['Time'] = data_original.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) 
data_original  = data_original[['SessionId', 'Time', 'ItemId']]

session_lens = data_original.groupby('SessionId')
session_lens = session_lens.size()

k = session_lens[session_lens>1]
data_original = data_original[np.in1d(data_original.SessionId, k.index)]

items = data_original.groupby('ItemId')
items = items.size()

k=items[items>=5]
data_original = data_original[np.in1d(data_original.ItemId, k.index)]

session_lens = data_original.groupby('SessionId')
session_lens = session_lens.size()

k= session_lens[session_lens>=2]
data_original = data_original[np.in1d(data_original.SessionId,k.index)]

max_time = data_original.Time.max()

session_times = data_original.groupby('SessionId').Time.max()
session_tr = session_times[session_times < max_time-86400].index
session_tt = session_times[session_times >= max_time-86400].index
tr = data_original[np.in1d(data_original['SessionId'], session_tr)]
tt = data_original[np.in1d(data_original['SessionId'], session_tt)]
tt = tt[np.in1d(tt['ItemId'], tr['ItemId'])]

l = tt.groupby('SessionId').size()
tt = tt[np.in1d(tt['SessionId'], l[l>=2].index)]

tr.to_csv(data_processed_path + 'rsc15_train_full.txt', sep='\t', index=False)
tt.to_csv(data_processed_path + 'rsc15_test.txt', sep='\t', index=False)

max_time = tr.Time.max()
session_times = tr.groupby('SessionId').Time.max()
session_tr = session_times[session_times < max_time-86400].index
session_val = session_times[session_times >= max_time-86400].index
train_tr = tr[np.in1d(tr['SessionId'], session_tr)]
val = tr[np.in1d(tr['SessionId'], session_val)]
val = val[np.in1d(val['ItemId'], train_tr.ItemId)]
l = val.groupby('SessionId').size()
val = val[np.in1d(val['SessionId'], l[l>=2].index)]

train_tr.to_csv(data_processed_path + 'rsc15_train_tr.txt', sep='\t', index=False)
val.to_csv(data_processed_path + 'rsc15_train_valid.txt', sep='\t', index=False)