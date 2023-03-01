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
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,BaggingRegressor
from sklearn.linear_model import LassoCV
from sklearn import linear_model
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


def parkes_error_zone_detailed(act, pred):

    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def parkes_type_1(act, pred):
                                
        # Zone E
        if above_line(0, 150/18, 35/18, 155/18) and above_line(35/18, 155/18, 50/18, 550/18):
            return 7
        # Zone D - left upper
        if (pred > 100/18 and above_line(25/18, 100/18, 50/18, 125/18) and
                above_line(50/18, 125/18, 80/18, 215/18) and above_line(80/18, 215/18, 125/18, 550/18)):
            return 6
        # Zone D - right lower
        if (act > 250/18 and below_line(250/18, 40/18, 550/18, 150/18)):
            return 5
        # Zone C - left upper
        if (pred > 60/18 and above_line(30/18, 60/18, 50/18, 80/18) and
                above_line(50/18, 80/18, 70/18, 110/18) and above_line(70/18, 110/18, 260/18, 550/18)):
            return 4
        # Zone C - right lower
        if (act > 120/18 and below_line(120/18, 30/18, 260/18, 130/18) and below_line(260/18, 130/18, 550/18, 250/18)):
            return 3
        # Zone B - left upper
        if (pred > 50/18 and above_line(30/18, 50/18, 140/18, 170/18) and
                above_line(140/18, 170/18, 280/18, 380/18) and (act < 280/18 or above_line(280/18, 380/18, 430/18, 550/18))):
            return 2
        # Zone B - right lower
        if (act > 50/18 and below_line(50/18, 30/18, 170/18, 145/18) and
                below_line(170/18, 145/18, 385/18, 300/18) and (act < 385/18 or below_line(385/18, 300/18, 550/18, 450/18))):
            return 1
        # Zone A
        return 0
    
    return parkes_type_1(act, pred)


parkes_error_zone_detailed = np.vectorize(parkes_error_zone_detailed)

def zone_accuracy(act_arr, pred_arr, detailed=False):

    acc = np.zeros(9)
    res = parkes_error_zone_detailed(act_arr, pred_arr)
   
    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]

    return acc / sum(acc)

def plot_CER(act, pred):
    plt.figure(dpi=500)
    plt.scatter(act, pred, marker='o', color='black', s=8)
    plt.title('Blood Glucose' + " Consensus Error Grid")
    plt.xlabel("Reference Glucose Concentration (mmol/L)")
    plt.ylabel("Prediction Glucose Concentration (mmol/L)")
    plt.xticks([0, 50/18, 100/18, 150/18, 200/18, 250/18, 300/18, 350/18, 400/18, 450/18, 500/18, 550/18])
    plt.yticks([0, 50/18, 100/18, 150/18, 200/18, 250/18, 300/18, 350/18, 400/18, 450/18, 500/18, 550/18])
    plt.gca().set_facecolor('white')
    plt.gca().set_xlim([0, 550/18])
    plt.gca().set_ylim([0, 550/18])
    plt.gca().set_aspect((550/18)/(550/18))
    plt.plot([0,550/18], [0,550/18], ':', c='black') 
    plt.plot([0, 35/18], [150/18, 155/18], '-', c='black')
    plt.plot([35/18, 50/18], [155/18, 550/18], '-', c='black') 
    plt.plot([25/18, 50/18], [100/18, 125/18], '-', c='black') 
    plt.plot([50/18, 80/18], [125/18, 215/18], '-', c='black') 
    plt.plot([80/18, 125/18], [215/18, 550/18], '-', c='black') 
    plt.plot([250/18, 550/18], [40/18, 150/18], '-', c='black') 
    plt.plot([30/18, 50/18], [60/18, 80/18], '-', c='blue')
    plt.plot([50/18, 70/18], [80/18, 110/18], '-', c='blue')
    plt.plot([70/18, 260/18], [110/18, 550/18], '-', c='blue')
    plt.plot([120/18, 260/18], [30/18, 130/18], '-', c='blue')
    plt.plot([260/18, 550/18], [130/18, 250/18], '-', c='blue')
    plt.plot([30/18, 140/18], [50/18, 170/18], '-', c='green')
    plt.plot([140/18, 280/18], [170/18, 380/18], '-', c='green')
    plt.plot([280/18, 430/18], [380/18,550/18], '-', c='green')
    plt.plot([50/18, 170/18], [30/18,145/18], '-', c='green')
    plt.plot([170/18, 385/18], [145/18,300/18], '-', c='green')
    plt.plot([385/18, 550/18], [300/18,450/18], '-', c='green')
    plt.plot([250/18, 250/18], [0,40/18], '-', c='black')
    plt.plot([120/18, 120/18], [0,30/18], '-', c='blue')
    plt.plot([50/18, 50/18], [0,30/18], '-', c='green')
    plt.plot([0, 30/18], [50/18,50/18], '-', c='green')
    plt.plot([0, 30/18], [60/18,60/18], '-', c='blue')
    plt.plot([0, 25/18], [100/18,100/18], '-', c='black')
    plt.text(15/18, 430/18, "E", fontsize=15)
    plt.text(60/18, 430/18, "D", fontsize=15)
    plt.text(140/18, 430/18, "C", fontsize=15)
    plt.text(230/18, 410/18, "B", fontsize=15, c='blue')
    plt.text(300/18, 360/18, "A", fontsize=15, c='green')
    plt.text(330/18, 290/18, "A", fontsize=15, c='green')
    plt.text(350/18, 200/18, "B", fontsize=15, c='blue')
    plt.text(360/18, 100/18, "C", fontsize=15)
    plt.text(370/18, 15/18, "D", fontsize=15)   
    
def RMSE_MARD(pred_values,ref_values):    
    data_length = len(ref_values)
    total = 0
    for i in range (data_length):
        temp = (ref_values[i] - pred_values[i]) * (ref_values[i] - pred_values[i])
        total = total + temp

    smse_value = math.sqrt(total / data_length)
    print('RMSE:{:.4f}'.format(smse_value))
    
    total = 0
    for i in range(data_length):
        temp = abs((ref_values[i] - pred_values[i]) / ref_values[i])
        total = total + temp
    mard_value = total / data_length
    print('MARD:{:.4f}'.format(mard_value))
        
    return smse_value,mard_value


def RMSE(pred_values,ref_values):    
    data_length = len(ref_values)
    total = 0
    for i in range (data_length):
        temp = (ref_values[i] - pred_values[i]) * (ref_values[i] - pred_values[i])
        total = total + temp
    smse_value = total / data_length 
    smse_value = smse_value * smse_value
    return smse_value

def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]

class Choquet_integral(torch.nn.Module):
    
    def __init__(self, N_in, N_out):
        super(Choquet_integral,self).__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.nVars = 2**self.N_in - 2
        
        dummy = (1./self.N_in) * torch.ones((self.nVars, self.N_out), requires_grad=True)
        self.vars = torch.nn.Parameter(dummy)
        
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.N_in)
        
        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]
        
    def forward(self,inputs):    
        self.FM = self.chi_nn_vars(self.vars)
        sortInputs, sortInd = torch.sort(inputs,1, True)
        M, N = inputs.size()
        sortInputs = torch.cat((sortInputs, torch.zeros(M,1)), 1)
        sortInputs = sortInputs[:,:-1] -  sortInputs[:,1:]
        
        out = torch.cumsum(torch.pow(2,sortInd),1) - torch.ones(1, dtype=torch.int64)
        
        data = torch.zeros((M,self.nVars+1))
        
        for i in range(M):
            data[i,out[i,:]] = sortInputs[i,:] 
        
        
        ChI = torch.matmul(data,self.FM)
            
        return ChI
    
    def chi_nn_vars(self, chi_vars):
        chi_vars = torch.abs(chi_vars)
        
        FM = chi_vars[None, 0,:]
        for i in range(1,self.nVars):
            indices = subset_to_indices(self.subset[i])
            if (len(indices) == 1):
                FM = torch.cat((FM,chi_vars[None,i,:]),0)
            else:
                maxVal,_ = torch.max(FM[indices,:],0)
                temp = torch.add(maxVal,chi_vars[i,:])
                FM = torch.cat((FM,temp[None,:]),0)
              
        FM = torch.cat([FM, torch.ones((1,self.N_out))],0)
        FM = torch.min(FM, torch.ones(1))  
        
        return FM
    
    
    
    

TS_train_feature = np.loadtxt('G:/BG/save/fold 1/temporal_train_feature.txt', delimiter=',')
TS_test_feature = np.loadtxt('G:/BG/save/fold 1/temporal_test_feature.txt', delimiter=',')
DP_train_feature = np.loadtxt('G:/BG/save/fold 1/morphological_train_feature.txt', delimiter=',')
DP_test_feature = np.loadtxt('G:/BG/save/fold 1/morphological_test_feature.txt', delimiter=',')

train_label = np.loadtxt('G:/BG/save/fold 1/train_glucose.txt', delimiter=',')
test_label = np.loadtxt('G:/BG/save/fold 1/test_glucose.txt', delimiter=',')

TS_DS_train = np.hstack((TS_train_feature, DP_train_feature))  
TS_DS_test = np.hstack((TS_test_feature, DP_test_feature))  


algo1 = RandomForestRegressor(n_estimators=200,max_samples = 0.85,ccp_alpha = 0.00001,min_samples_leaf=2,max_leaf_nodes = 100,
                              criterion='mse',
                              random_state=30,
                              max_depth=40,min_samples_split=2,
                              n_jobs=100)
    
algo1.fit(TS_DS_train, train_label) 
train_pred1_1 = algo1.predict(TS_DS_train)
# train_result1_1 = RMSE_MARD(train_pred1_1,train_label)
test_pred1_1 = algo1.predict(TS_DS_test)
test_result1_1 = RMSE_MARD(test_pred1_1,test_label)
plot_show = plot_CER(test_label, test_pred1_1)

algo1.fit(DP_train_feature, train_label) 
train_pred1_2 = algo1.predict(DP_train_feature)
# train_result1_2 = RMSE_MARD(train_pred1_2,train_label)
test_pred1_2 = algo1.predict(DP_test_feature)
test_result1_2 = RMSE_MARD(test_pred1_2,test_label)

algo1.fit(TS_train_feature, train_label) 
train_pred1_3 = algo1.predict(TS_train_feature)
# train_result1_3 = RMSE_MARD(train_pred1_3,train_label)
test_pred1_3 = algo1.predict(TS_test_feature)
test_result1_3 = RMSE_MARD(test_pred1_3,test_label)
print("-----------------------------------")



algo2 = GradientBoostingRegressor()
    
algo2.fit(TS_DS_train, train_label) 
train_pred2_1 = algo2.predict(TS_DS_train)
# train_result2_1 = RMSE_MARD(train_pred2_1,train_label)
test_pred2_1 = algo2.predict(TS_DS_test)
test_result2_1 = RMSE_MARD(test_pred2_1,test_label)

algo2.fit(DP_train_feature, train_label) 
train_pred2_2 = algo2.predict(DP_train_feature)
# train_result2_2 = RMSE_MARD(train_pred2_2,train_label)
test_pred2_2 = algo2.predict(DP_test_feature)
test_result2_2 = RMSE_MARD(test_pred2_2,test_label)

algo2.fit(TS_train_feature, train_label) 
train_pred2_3 = algo2.predict(TS_train_feature)
# train_result2_3 = RMSE_MARD(train_pred2_3,train_label)
test_pred2_3 = algo2.predict(TS_test_feature)
test_result2_3 = RMSE_MARD(test_pred2_3,test_label)
print("-----------------------------------")




algo3 = BaggingRegressor(n_estimators = 20)   
algo3.fit(TS_DS_train, train_label) 
train_pred3_1 = algo3.predict(TS_DS_train)
# train_result3_1 = RMSE_MARD(train_pred3_1,train_label)
test_pred3_1 = algo3.predict(TS_DS_test)
test_result3_1 = RMSE_MARD(test_pred3_1,test_label)


algo3.fit(DP_train_feature, train_label) 
train_pred3_2 = algo3.predict(DP_train_feature)
# train_result3_2 = RMSE_MARD(train_pred3_2,train_label)
test_pred3_2 = algo3.predict(DP_test_feature)
test_result3_2 = RMSE_MARD(test_pred3_2,test_label)


algo3.fit(TS_train_feature, train_label) 
train_pred3_3 = algo3.predict(TS_train_feature)
# train_result3_3 = RMSE_MARD(train_pred3_3,train_label)
test_pred3_3 = algo3.predict(TS_test_feature)
test_result3_3 = RMSE_MARD(test_pred3_3,test_label)
print("-----------------------------------")



N = 7
num_train = len(train_label) // N
total_train = int(num_train * N)
Train_Pred1_1_list = []
Train_Pred1_2_list = []
Train_Pred1_3_list = []
Train_Pred2_1_list = []
Train_Pred2_2_list = []
Train_Pred2_3_list = []
Train_Pred3_1_list = []
Train_Pred3_2_list = []
Train_Pred3_3_list = []
Train_Label_list = []

for m in range(total_train - N):
    temp = train_pred1_1[m:m+N].reshape(1,-1)
    Train_Pred1_1_list.append(temp)
    temp = train_pred1_2[m:m+N].reshape(1,-1)
    Train_Pred1_2_list.append(temp)
    temp = train_pred1_3[m:m+N].reshape(1,-1)
    Train_Pred1_3_list.append(temp)
    temp = train_pred2_1[m:m+N].reshape(1,-1)
    Train_Pred2_1_list.append(temp)
    temp = train_pred2_2[m:m+N].reshape(1,-1)
    Train_Pred2_2_list.append(temp)
    temp = train_pred2_3[m:m+N].reshape(1,-1)
    Train_Pred2_3_list.append(temp)
    temp = train_pred3_1[m:m+N].reshape(1,-1)
    Train_Pred3_1_list.append(temp)
    temp = train_pred3_2[m:m+N].reshape(1,-1)
    Train_Pred3_2_list.append(temp)
    temp = train_pred3_3[m:m+N].reshape(1,-1)
    Train_Pred3_3_list.append(temp)    
    Train_Label_list.append(train_label[m+N-1])
    
Train_Pred1_1 = np.array(Train_Pred1_1_list)
Train_Pred1_1 = Train_Pred1_1.reshape(-1,N)
Train_Pred1_2 = np.array(Train_Pred1_2_list)
Train_Pred1_2 = Train_Pred1_2.reshape(-1,N)
Train_Pred1_3 = np.array(Train_Pred1_3_list)
Train_Pred1_3 = Train_Pred1_3.reshape(-1,N)

Train_Pred2_1 = np.array(Train_Pred2_1_list)
Train_Pred2_1 = Train_Pred2_1.reshape(-1,N)
Train_Pred2_2 = np.array(Train_Pred2_2_list)
Train_Pred2_2 = Train_Pred2_2.reshape(-1,N)
Train_Pred2_3 = np.array(Train_Pred2_3_list)
Train_Pred2_3 = Train_Pred2_3.reshape(-1,N)

Train_Pred3_1 = np.array(Train_Pred3_1_list)
Train_Pred3_1 = Train_Pred3_1.reshape(-1,N)
Train_Pred3_2 = np.array(Train_Pred3_2_list)
Train_Pred3_2 = Train_Pred3_2.reshape(-1,N)
Train_Pred3_3 = np.array(Train_Pred3_3_list)
Train_Pred3_3 = Train_Pred3_3.reshape(-1,N)

Train_Label = np.array(Train_Label_list)

num_test = len(test_label) // N
total_test = int(num_test * N)
Test_Pred1_1_list = []
Test_Pred1_2_list = []
Test_Pred1_3_list = []
Test_Pred2_1_list = []
Test_Pred2_2_list = []
Test_Pred2_3_list = []
Test_Pred3_1_list = []
Test_Pred3_2_list = []
Test_Pred3_3_list = []
Test_Label_list = []

for m in range(total_test - N):
    temp = test_pred1_1[m:m+N].reshape(1,-1)
    Test_Pred1_1_list.append(temp)
    temp = test_pred1_2[m:m+N].reshape(1,-1)
    Test_Pred1_2_list.append(temp)
    temp = test_pred1_3[m:m+N].reshape(1,-1)
    Test_Pred1_3_list.append(temp)
    temp = test_pred2_1[m:m+N].reshape(1,-1)
    Test_Pred2_1_list.append(temp)
    temp = test_pred2_2[m:m+N].reshape(1,-1)
    Test_Pred2_2_list.append(temp)
    temp = test_pred2_3[m:m+N].reshape(1,-1)
    Test_Pred2_3_list.append(temp)
    temp = test_pred3_1[m:m+N].reshape(1,-1)
    Test_Pred3_1_list.append(temp)
    temp = test_pred3_2[m:m+N].reshape(1,-1)
    Test_Pred3_2_list.append(temp)
    temp = test_pred3_3[m:m+N].reshape(1,-1)
    Test_Pred3_3_list.append(temp)    
    Test_Label_list.append(test_label[m+N-1])
    
Test_Pred1_1 = np.array(Test_Pred1_1_list)
Test_Pred1_1 = Test_Pred1_1.reshape(-1,N)
Test_Pred1_2 = np.array(Test_Pred1_2_list)
Test_Pred1_2 = Test_Pred1_2.reshape(-1,N)
Test_Pred1_3 = np.array(Test_Pred1_3_list)
Test_Pred1_3 = Test_Pred1_3.reshape(-1,N)

Test_Pred2_1 = np.array(Test_Pred2_1_list)
Test_Pred2_1 = Test_Pred2_1.reshape(-1,N)
Test_Pred2_2 = np.array(Test_Pred2_2_list)
Test_Pred2_2 = Test_Pred2_2.reshape(-1,N)
Test_Pred2_3 = np.array(Test_Pred2_3_list)
Test_Pred2_3 = Test_Pred2_3.reshape(-1,N)

Test_Pred3_1 = np.array(Test_Pred3_1_list)
Test_Pred3_1 = Test_Pred3_1.reshape(-1,N)
Test_Pred3_2 = np.array(Test_Pred3_2_list)
Test_Pred3_2 = Test_Pred3_2.reshape(-1,N)
Test_Pred3_3 = np.array(Test_Pred3_3_list)
Test_Pred3_3 = Test_Pred3_3.reshape(-1,N)

Test_Label = np.array(Test_Label_list)



Train_Pred1_1 = torch.tensor(Train_Pred1_1,dtype=torch.float)
Train_Pred1_2 = torch.tensor(Train_Pred1_2,dtype=torch.float)
Train_Pred1_3 = torch.tensor(Train_Pred1_3,dtype=torch.float)
Train_Pred2_1 = torch.tensor(Train_Pred2_1,dtype=torch.float)
Train_Pred2_2 = torch.tensor(Train_Pred2_2,dtype=torch.float)
Train_Pred2_3 = torch.tensor(Train_Pred2_3,dtype=torch.float)
Train_Pred3_1 = torch.tensor(Train_Pred3_1,dtype=torch.float)
Train_Pred3_2 = torch.tensor(Train_Pred3_2,dtype=torch.float)
Train_Pred3_3 = torch.tensor(Train_Pred3_3,dtype=torch.float)
Train_ID = torch.tensor(Train_Label,dtype=torch.float)

Test_Pred1_1 = torch.tensor(Test_Pred1_1,dtype=torch.float)
Test_Pred1_2 = torch.tensor(Test_Pred1_2,dtype=torch.float)
Test_Pred1_3 = torch.tensor(Test_Pred1_3,dtype=torch.float)
Test_Pred2_1 = torch.tensor(Test_Pred2_1,dtype=torch.float)
Test_Pred2_2 = torch.tensor(Test_Pred2_2,dtype=torch.float)
Test_Pred2_3 = torch.tensor(Test_Pred2_3,dtype=torch.float)
Test_Pred3_1 = torch.tensor(Test_Pred3_1,dtype=torch.float)
Test_Pred3_2 = torch.tensor(Test_Pred3_2,dtype=torch.float)
Test_Pred3_3 = torch.tensor(Test_Pred3_3,dtype=torch.float)
Test_ID = torch.tensor(Test_Label,dtype=torch.float)



net1_1 = Choquet_integral(N_in=N,N_out=1)  
net1_2 = Choquet_integral(N_in=N,N_out=1) 
net1_3 = Choquet_integral(N_in=N,N_out=1) 
net2_1 = Choquet_integral(N_in=N,N_out=1)  
net2_2 = Choquet_integral(N_in=N,N_out=1) 
net2_3 = Choquet_integral(N_in=N,N_out=1) 
net3_1 = Choquet_integral(N_in=N,N_out=1)  
net3_2 = Choquet_integral(N_in=N,N_out=1) 
net3_3 = Choquet_integral(N_in=N,N_out=1) 

  
learning_rate = 0.001
criterion1_1 = torch.nn.MSELoss(reduction='mean')
optimizer1_1 = torch.optim.Adam(net1_1.parameters(), lr=learning_rate)  
criterion1_2 = torch.nn.MSELoss(reduction='mean')
optimizer1_2 = torch.optim.Adam(net1_2.parameters(), lr=learning_rate) 
criterion1_3 = torch.nn.MSELoss(reduction='mean')
optimizer1_3 = torch.optim.Adam(net1_3.parameters(), lr=learning_rate) 

criterion2_1 = torch.nn.MSELoss(reduction='mean')
optimizer2_1 = torch.optim.Adam(net2_1.parameters(), lr=learning_rate)  
criterion2_2 = torch.nn.MSELoss(reduction='mean')
optimizer2_2 = torch.optim.Adam(net2_2.parameters(), lr=learning_rate) 
criterion2_3 = torch.nn.MSELoss(reduction='mean')
optimizer2_3 = torch.optim.Adam(net2_3.parameters(), lr=learning_rate) 

criterion3_1 = torch.nn.MSELoss(reduction='mean')
optimizer3_1 = torch.optim.Adam(net3_1.parameters(), lr=learning_rate)  
criterion3_2 = torch.nn.MSELoss(reduction='mean')
optimizer3_2 = torch.optim.Adam(net3_2.parameters(), lr=learning_rate) 
criterion3_3 = torch.nn.MSELoss(reduction='mean')
optimizer3_3 = torch.optim.Adam(net3_3.parameters(), lr=learning_rate) 
     
num_epochs = 10
        
for t in range(num_epochs):
    print('Epoch: %d' %(t + 1))
    optimizer1_1.zero_grad()    
    Train_P1_1 = net1_1(Train_Pred1_1)
    Train_P1_1 = Train_P1_1.view(-1)
    loss1_1 = criterion1_1(Train_P1_1, Train_ID)
    loss1_1.backward()
    optimizer1_1.step()     
    Train_P1_1 = Train_P1_1.cpu().detach().numpy()   
    train_result1_1 = RMSE(Train_P1_1, Train_Label) 
    P1_1 = 1 / train_result1_1
    
    optimizer1_2.zero_grad()    
    Train_P1_2 = net1_2(Train_Pred1_2)
    Train_P1_2 = Train_P1_2.view(-1)
    loss1_2 = criterion1_2(Train_P1_2, Train_ID)
    loss1_2.backward()
    optimizer1_2.step()     
    Train_P1_2 = Train_P1_2.cpu().detach().numpy()   
    train_result1_2 = RMSE(Train_P1_2, Train_Label) 
    P1_2 = 1 / train_result1_2

    Train_P1_3 = net1_3(Train_Pred1_3)
    Train_P1_3 = Train_P1_3.view(-1)
    loss1_3 = criterion1_3(Train_P1_3, Train_ID)
    loss1_3.backward()
    optimizer1_3.step()     
    Train_P1_3 = Train_P1_3.cpu().detach().numpy()   
    train_result1_3 = RMSE(Train_P1_3, Train_Label) 
    P1_3 = 1 / train_result1_3


    optimizer2_1.zero_grad()    
    Train_P2_1 = net2_1(Train_Pred2_1)
    Train_P2_1 = Train_P2_1.view(-1)
    loss2_1 = criterion2_1(Train_P2_1, Train_ID)
    loss2_1.backward()
    optimizer2_1.step()     
    Train_P2_1 = Train_P2_1.cpu().detach().numpy()   
    train_result2_1 = RMSE(Train_P2_1, Train_Label) 
    P2_1 = 1 / train_result2_1

    optimizer2_2.zero_grad()    
    Train_P2_2 = net2_2(Train_Pred2_2)
    Train_P2_2 = Train_P2_2.view(-1)
    loss2_2 = criterion2_2(Train_P2_2, Train_ID)
    loss2_2.backward()
    optimizer2_2.step()     
    Train_P2_2 = Train_P2_2.cpu().detach().numpy()   
    train_result2_2 = RMSE(Train_P2_2, Train_Label) 
    P2_2 = 1 / train_result2_2
    
    optimizer2_3.zero_grad()    
    Train_P2_3 = net2_3(Train_Pred2_3)
    Train_P2_3 = Train_P2_3.view(-1)
    loss2_3 = criterion2_3(Train_P2_3, Train_ID)
    loss2_3.backward()
    optimizer2_3.step()     
    Train_P2_3 = Train_P2_3.cpu().detach().numpy()   
    train_result2_3 = RMSE(Train_P2_3, Train_Label) 
    P2_3 = 1 / train_result2_3


    optimizer3_1.zero_grad()    
    Train_P3_1 = net3_1(Train_Pred3_1)
    Train_P3_1 = Train_P3_1.view(-1)
    loss3_1 = criterion3_1(Train_P3_1, Train_ID)
    loss3_1.backward()
    optimizer3_1.step()     
    Train_P3_1 = Train_P3_1.cpu().detach().numpy()   
    train_result3_1 = RMSE(Train_P3_1, Train_Label) 
    P3_1 = 1 / train_result3_1
    
    optimizer3_2.zero_grad()    
    Train_P3_2 = net3_2(Train_Pred3_2)
    Train_P3_2 = Train_P3_2.view(-1)
    loss3_2 = criterion3_2(Train_P3_2, Train_ID)
    loss3_2.backward()
    optimizer3_2.step()     
    Train_P3_2 = Train_P3_2.cpu().detach().numpy()   
    train_result3_2 = RMSE(Train_P3_2, Train_Label) 
    P3_2 = 1 / train_result3_2

    optimizer3_3.zero_grad()    
    Train_P3_3 = net3_3(Train_Pred3_3)
    Train_P3_3 = Train_P3_3.view(-1)
    loss3_3 = criterion3_3(Train_P3_3, Train_ID)
    loss3_3.backward()
    optimizer3_3.step()     
    Train_P3_3 = Train_P3_3.cpu().detach().numpy()   
    train_result3_3 = RMSE(Train_P3_3, Train_Label) 
    P3_3 = 1 / train_result3_3    

 

    net1_1.eval()   
    net1_2.eval()  
    net1_3.eval()  
    net2_1.eval()   
    net2_2.eval()  
    net2_3.eval()  
    net3_1.eval()   
    net3_2.eval()  
    net3_3.eval()     
 
    
    test_1_1 = net1_1(Test_Pred1_1)
    test_1_1 = test_1_1.cpu().detach().numpy()
    test_1_1 = test_1_1.reshape(-1)
    test_1_2 = net1_2(Test_Pred1_2)
    test_1_2 = test_1_2.cpu().detach().numpy()
    test_1_2 = test_1_2.reshape(-1)
    test_1_3 = net1_3(Test_Pred1_3)
    test_1_3 = test_1_3.cpu().detach().numpy()
    test_1_3 = test_1_3.reshape(-1)

    test_2_1 = net2_1(Test_Pred2_1)
    test_2_1 = test_2_1.cpu().detach().numpy()
    test_2_1 = test_2_1.reshape(-1)
    test_2_2 = net2_2(Test_Pred2_2)
    test_2_2 = test_2_2.cpu().detach().numpy()
    test_2_2 = test_2_2.reshape(-1)
    test_2_3 = net2_3(Test_Pred2_3)
    test_2_3 = test_2_3.cpu().detach().numpy()
    test_2_3 = test_2_3.reshape(-1)

    test_3_1 = net3_1(Test_Pred3_1)
    test_3_1 = test_3_1.cpu().detach().numpy()
    test_3_1 = test_3_1.reshape(-1)
    test_3_2 = net3_2(Test_Pred3_2)
    test_3_2 = test_3_2.cpu().detach().numpy()
    test_3_2 = test_3_2.reshape(-1)
    test_3_3 = net3_3(Test_Pred3_3)
    test_3_3 = test_3_3.cpu().detach().numpy()
    test_3_3 = test_3_3.reshape(-1)
    
    P_all = P1_1 + P1_2 + P1_3 + P2_1 + P2_2 + P2_3 + P3_1 + P3_2 + P3_3
    test_fusion = (
        (P1_1 / P_all)*test_1_1 + (P1_2 / P_all)*test_1_2 +(P1_3 / P_all)*test_1_3) + (
        (P2_1 / P_all)*test_2_1 + (P2_2 / P_all)*test_2_2 +(P2_3 / P_all)*test_2_3) + (
        (P3_1 / P_all)*test_3_1 + (P3_2 / P_all)*test_3_2 +(P3_3 / P_all)*test_3_3)
    
    fusion_result = RMSE_MARD(test_fusion, Test_Label) 

    
    result = zone_accuracy(Test_Label, test_fusion, detailed=False)
    print('Zone A is %1.4f, Zone B is %1.4f'% (result[0]*100, result[1]*100))
    print('Zone A + Zone B is :{:.2f} %'.format((result[0]+result[1])*100))
    plot_show = plot_CER(Test_Label, test_fusion)
    print("-----------------------------------")
    
    
    
# FM_learned1 = (net1.chi_nn_vars(net1.vars).cpu()).detach().numpy()
# FM_learned2 = (net2.chi_nn_vars(net2.vars).cpu()).detach().numpy()
# FM_learned3 = (net3.chi_nn_vars(net3.vars).cpu()).detach().numpy()
  
       










