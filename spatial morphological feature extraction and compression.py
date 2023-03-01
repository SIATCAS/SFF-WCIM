import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
import torch,re,os,time
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split
from sklearn.utils import shuffle
import scipy.io as scio
import torch.nn as nn
import gc, math 
from sklearn.ensemble import RandomForestRegressor
torch.__version__



torch.manual_seed(128)
np.random.seed(1024)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

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



class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, p1, p2, p3, p4):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,kernel_size=(1,1),stride=(1,2), padding=p1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=p2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,kernel_size=(1,1),stride=1, padding=p3, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, kernel_size=(1,1), stride=(1,2), padding=p4, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X     
    
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters, p1, p2, p3):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,kernel_size=(1,1),stride=1, padding=p1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=p2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,kernel_size=(1,1),stride=1, padding=p3, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X
    
class ResModel(nn.Module):
    def __init__(self):
        super(ResModel,self).__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels =1,out_channels =4,kernel_size=(1,6),stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d((1,2),padding=0),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(4, f=(1,6), filters=[4, 4, 16],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(16, f=(1,6), filters=[8, 8, 32],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(32, f=(1,6), filters=[16, 16, 64],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(64, f=(1,6), filters=[32, 32, 128],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
        )
        
        self.stage6 = nn.Sequential(
            nn.Conv2d(in_channels =1,out_channels =4,kernel_size=(1,6),stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d((1,2),padding=0),
        )
        self.stage7 = nn.Sequential(
            ConvBlock(4, f=(1,6), filters=[4, 4, 16],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage8 = nn.Sequential(
            ConvBlock(16, f=(1,6), filters=[8, 8, 32],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage9 = nn.Sequential(
            ConvBlock(32, f=(1,6), filters=[16, 16, 64],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage10 = nn.Sequential(
            ConvBlock(64, f=(1,6), filters=[32, 32, 128],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
        )   
        
        self.bn = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d((1,2),padding=0)
        
        self.fc1 = nn.Linear(66560,4000)
        self.fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(4000,400)
        self.fc4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(400,40)
        self.fc6 = nn.Linear(40,1)
        
       
        
    
    def forward(self, X, Y):
        out1 = self.stage1(X)
        out1 = self.stage2(out1)
        out1 = self.stage3(out1)
        out1 = self.stage4(out1)
        out1 = self.stage5(out1)
        out1 = self.pool(out1)   
        out1 = self.bn(out1)
        out1 = out1.view(out1.size(0),-1)
        
        out2 = self.stage6(Y)
        out2 = self.stage7(out2)
        out2 = self.stage8(out2)
        out2 = self.stage9(out2)
        out2 = self.stage10(out2)
        out2 = self.pool(out2)   
        out2 = self.bn(out2)
        out2 = out2.view(out2.size(0),-1)
        
        out_p1 = torch.cat((out1,out2),dim = 1)          
        out_p2 = self.fc1(out_p1)
        out_p3 = self.fc2(out_p2)
        out_p4 = self.fc3(out_p3)
        out_p5 = self.fc4(out_p4)
        out_p6 = self.fc5(out_p5)
        out_p7 = self.fc6(out_p6)
        
        return out_p6
    


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

train_data_ECG = np.zeros((1,760))
test_data_ECG = np.zeros((1,760))
train_data_PPG = np.zeros((1,760))
test_data_PPG = np.zeros((1,760))
train_label_list = []
test_label_list = []

path1 = 'G:/BG/ECG_norm'
path2 = 'G:/BG/PPG_norm'
data_dir_list1 = sorted(os.listdir(path1),key=numericalSort) 
data_dir_s1 = shuffle(data_dir_list1, random_state = 8)
data_dir_list2 = sorted(os.listdir(path2),key=numericalSort) 
data_dir_s2 = shuffle(data_dir_list2, random_state = 8)


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
for path in data_dir_s1:
    if  N > 9:
      all_data = scio.loadmat(path1 + '/' + path)   
      temp1 = all_data["save_ECG"]
      temp_data = temp1[:,0:760]
      temp_label = temp1[:,760]
      train_data_ECG = np.vstack((train_data_ECG,temp_data)) 
      K = temp_data.shape[0]
      for k in range(K):
          if k % 20 == 0:
              train_label_list.append(temp_label[k])
    else:
      all_data = scio.loadmat(path1 + '/' + path)   
      temp1 = all_data["save_ECG"]
      temp_data = temp1[:,0:760]
      temp_label = temp1[:,760]
      test_data_ECG = np.vstack((test_data_ECG,temp_data))
      K = temp_data.shape[0]
      for k in range(K):   
          if k % 20 == 0:
              test_label_list.append(temp_label[k])        
    N = N + 1 

train_data_ECG = np.delete(train_data_ECG,[0],axis=0) 
train_data_ECG = train_data_ECG.reshape(-1,20,760)

test_data_ECG = np.delete(test_data_ECG,[0],axis=0)
test_data_ECG = test_data_ECG.reshape(-1,20,760)

train_label = np.array(train_label_list)
test_label = np.array(test_label_list)



N = 0
for path in data_dir_s2:
    if N > 9:
      all_data = scio.loadmat(path2 + '/' + path)   
      temp2 = all_data["save_PPG"]
      temp_data = temp2[:,0:760]
      temp_label = temp2[:,760]
      train_data_PPG = np.vstack((train_data_PPG,temp_data))  

    else:
      all_data = scio.loadmat(path2 + '/' + path)   
      temp2 = all_data["save_PPG"]
      temp_data = temp2[:,0:760]
      temp_label = temp2[:,760]
      test_data_PPG = np.vstack((test_data_PPG,temp_data))       
    N = N + 1 

train_data_PPG = np.delete(train_data_PPG,[0],axis=0) 
train_data_PPG = train_data_PPG.reshape(-1,20,760)

test_data_PPG = np.delete(test_data_PPG,[0],axis=0)
test_data_PPG = test_data_PPG.reshape(-1,20,760)


train_ID = np.arange(0,len(train_label),1)
test_ID = np.arange(0,len(test_label),1)
train_ID = torch.from_numpy(train_ID)
test_ID = torch.from_numpy(test_ID)


train_Label = torch.from_numpy(train_label)
test_Label = torch.from_numpy(test_label)


BATCH_SIZE = 128
train_set = torch.utils.data.TensorDataset(train_ID, train_Label)
test_set = torch.utils.data.TensorDataset(test_ID, test_Label)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False) 
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net = torch.load("G:/BG/save/fold 1/resmodel_EPOCHS 10.pth")
net.eval() 


train_feature = np.zeros((1,40))
test_feature = np.zeros((1,40))
       
for m, (train_images, tr_labels) in enumerate(train_loader):
    train_images = train_images.numpy()
    data_len = len(train_images)
    train_ECG_list = []
    train_PPG_list = []
    for n in range(data_len):
        path_temp = train_images[n]
        temp_ECG = train_data_ECG[path_temp]
        temp_PPG = train_data_PPG[path_temp]
        train_ECG_list.append(temp_ECG)
        train_PPG_list.append(temp_PPG)
        
    train_ECG = np.array(train_ECG_list)
    train_ECG = np.expand_dims(train_ECG, axis=1) 
    train_ECG = torch.from_numpy(train_ECG)
    train_ECG = train_ECG.float().to(DEVICE)        

    train_PPG = np.array(train_PPG_list)
    train_PPG = np.expand_dims(train_PPG, axis=1) 
    train_PPG = torch.from_numpy(train_PPG)
    train_PPG = train_PPG.float().to(DEVICE)        
    train_outputs = net(train_ECG,train_PPG)
                     
    train_outputs = train_outputs.cpu()
    train_outputs = train_outputs.detach().numpy()
    train_feature = np.vstack((train_feature,train_outputs))    
train_feature = np.delete(train_feature,[0],axis=0)    
    
  
for n, (test_images, te_labels) in enumerate(test_loader):
    test_images = test_images.numpy()
    data_len = len(test_images)
    test_ECG_list = []
    test_PPG_list = []
    for n in range(data_len):
        path_temp = test_images[n]
        temp_ECG = test_data_ECG[path_temp]
        temp_PPG = test_data_PPG[path_temp]
        test_ECG_list.append(temp_ECG)
        test_PPG_list.append(temp_PPG)
        
    test_ECG = np.array(test_ECG_list)
    test_ECG = np.expand_dims(test_ECG, axis=1) 
    test_ECG = torch.from_numpy(test_ECG)
    test_ECG = test_ECG.float().to(DEVICE)        

    test_PPG = np.array(test_PPG_list)
    test_PPG = np.expand_dims(test_PPG, axis=1) 
    test_PPG = torch.from_numpy(test_PPG)
    test_PPG = test_PPG.float().to(DEVICE)        

    test_outputs = net(test_ECG,test_PPG)              
    test_outputs = test_outputs.cpu()
    test_outputs = test_outputs.detach().numpy()
    test_feature = np.vstack((test_feature,test_outputs)) 
test_feature = np.delete(test_feature,[0],axis=0)     


np.savetxt('G:/BG/save/fold 1/morphological_train_feature.txt', train_feature, delimiter=',', fmt='%1.5f')
np.savetxt('G:/BG/save/fold 1/morphological_test_feature.txt', test_feature, delimiter=',', fmt='%1.5f')
np.savetxt('G:/BG/save/fold 1/train_glucose.txt', train_label, delimiter=',', fmt='%1.2f')
np.savetxt('G:/BG/save/fold 1/test_glucose.txt', test_label, delimiter=',', fmt='%1.2f')




