import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler 
import torch,re,os,time
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split
from sklearn.utils import shuffle
import scipy.io as scio
torch.__version__
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression,f_regression
from sklearn.feature_selection import SelectKBest 
import math
torch.manual_seed(128)
np.random.seed(1024)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma



numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

train_base = np.zeros((1,161))
test_base = np.zeros((1,161))
path1 = 'G:/BG/ECG_PPG'
data_dir_list = sorted(os.listdir(path1),key=numericalSort) 
data_dir_s = shuffle(data_dir_list, random_state = 8)


"""
fold 1:  N > 9
fold 2:  N < 10 or N > 19
fold 3:  N < 20 or N > 29
fold 4:  N < 30 or N > 39
fold 5:  N < 40 or N > 49
fold 6:  N < 50 or N > 59
fold 7:  N < 60 or N > 69
fold 8:  N < 70 or N > 79
fold 9:  N < 80 or N > 89
fold 10:  N < 90 or N > 99
"""

N = 0
for path in data_dir_s:
    if  N > 9:
      all_data = scio.loadmat(path1 + '/' + path)   
      temp1 = all_data["feature_temp"]
      temp1 = np.real(temp1)
      train_base = np.vstack((train_base,temp1))   
    else:
      all_data = scio.loadmat(path1 + '/' + path)   
      temp1 = all_data["feature_temp"]
      temp1 = np.real(temp1)
      test_base = np.vstack((test_base,temp1))         
    N = N + 1 

train_base = np.delete(train_base,[0],axis=0) 
test_base = np.delete(test_base,[0],axis=0)

train_data = train_base[:,0:160] 
train_label = train_base[:,160]
test_data = test_base[:,0:160] 
test_label = test_base[:,160]

test_len = len(test_label)




train_data, train_label = shuffle(train_data, train_label,random_state=16)
feature_num = 80
train_feature1 = SelectKBest(f_regression, k=feature_num).fit_transform(train_data,train_label)
sect_label1_list = []

for m in range(feature_num):
    temp_feature = train_feature1[:,m]
    for n in range(160):
        set_feature = train_data[:,n]
        if (temp_feature[0] == set_feature[0] and temp_feature[1] == set_feature[1] and temp_feature[2] == set_feature[2] and
            temp_feature[3] == set_feature[3] and temp_feature[4] == set_feature[4] and temp_feature[5] == set_feature[5] and 
            temp_feature[6] == set_feature[6] and temp_feature[7] == set_feature[7] and temp_feature[8] == set_feature[8] and
            temp_feature[9] == set_feature[9] and temp_feature[10] == set_feature[10]):
                
            sect_label1_list.append(n) 
            break    
        
sect_label1 = np.array(sect_label1_list)


from sklearn.feature_selection import RFE
selector = RFE(estimator=RandomForestRegressor(n_estimators=200,max_samples = 0.87,ccp_alpha = 0.0002,min_samples_leaf=2,
                max_leaf_nodes = 30, criterion='mse', random_state=30,
                              max_depth=40,min_samples_split=4,
                              n_jobs=100),n_features_to_select=feature_num)
train_feature2 = selector.fit_transform(train_data,train_label)
sect_label2_list = []

for m in range(feature_num):
    temp_feature = train_feature2[:,m]
    for n in range(160):
        set_feature = train_data[:,n]
        if (temp_feature[0] == set_feature[0] and temp_feature[1] == set_feature[1] and temp_feature[2] == set_feature[2] and
            temp_feature[3] == set_feature[3] and temp_feature[4] == set_feature[4] and temp_feature[5] == set_feature[5] and 
            temp_feature[6] == set_feature[6] and temp_feature[7] == set_feature[7] and temp_feature[8] == set_feature[8] and
            temp_feature[9] == set_feature[9] and temp_feature[10] == set_feature[10]):
                
            sect_label2_list.append(n) 
            break    
        
sect_label2 = np.array(sect_label2_list)


from sklearn.feature_selection import SelectFromModel
las = LassoCV(cv= 5, n_alphas =100, eps =0.000001, tol = 0.000001, max_iter = 30000).fit(train_data, train_label) 
sfm = SelectFromModel(las, prefit=True,max_features = feature_num, threshold = -np.inf)
train_feature3 = sfm.transform(train_data)

sect_label3_list = []

for m in range(feature_num):
    temp_feature = train_feature3[:,m]
    for n in range(160):
        set_feature = train_data[:,n]
        if (temp_feature[0] == set_feature[0] and temp_feature[1] == set_feature[1] and temp_feature[2] == set_feature[2] and
            temp_feature[3] == set_feature[3] and temp_feature[4] == set_feature[4] and temp_feature[5] == set_feature[5] and 
            temp_feature[6] == set_feature[6] and temp_feature[7] == set_feature[7] and temp_feature[8] == set_feature[8] and
            temp_feature[9] == set_feature[9] and temp_feature[10] == set_feature[10]):
                
            sect_label3_list.append(n) 
            break    
        
sect_label3 = np.array(sect_label3_list)

com = list(set(sect_label1).intersection(sect_label2,sect_label3))
com_result = np.array(com).astype(int)
com_result_sort = np.sort(com_result)



