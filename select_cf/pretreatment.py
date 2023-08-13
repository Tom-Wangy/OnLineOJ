import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable
from hyperparameter import args
from model import Ynet
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils import round_pred
# from utils import get_info,Decode,get_normalization
import  utils
import matplotlib.pyplot as plt
import time
from skrebate import ReliefF
# import ACGA

class Data_loader:
    def __init__(self,number,args) -> None:
        args.data = '/root/C++_program/OnlineOJ/oj_server/data/'+str(number)+'/data.txt'
        self.get_data(args)
        self.get_weight()
        self.get_loader()
        self.get_max_paramters()
        pass

    def get_data(self,args):
             
        data_o = pd.read_csv(args.data, header=None)  # 原始特征
        self.X = np.array(data_o,dtype=np.float64)[:,0:-1]
        self.y = np.array(data_o)[:,-1]
        if(np.min(self.y) != 0):
            self.y = self.y-1

        min_max_scaler = preprocessing.StandardScaler()
        self.data = min_max_scaler.fit_transform(self.X)
        self.label = self.y
        pass

    def get_weight(self):
        relief = ReliefF(n_neighbors=20)
        # 使用ReliefF算法进行特征选择
        feature_weights = relief.fit(self.X, self.y).feature_importances_
        print(feature_weights.tolist())

        self.weight = utils.get_normalization(np.abs(feature_weights))
        pass

    def get_loader(self):
        data_train, data_valid, label_train, label_valid = train_test_split(self.data, self.label, test_size=0.2,stratify=self.label,random_state=42)
        data_train, data_test, label_train, label_test = train_test_split(data_train, label_train, test_size=2/8,stratify=label_train,random_state=42)

        X_train = torch.from_numpy(np.array(data_train)).float()
        X_valid = torch.from_numpy(np.array(data_valid)).float()
        y_train = torch.from_numpy(np.array(label_train)).long()
        y_valid = torch.from_numpy(np.array(label_valid)).long()
        X_test = torch.from_numpy(data_test).float()
        y_test = torch.from_numpy(label_test).long()

        self.train_loader = Data.DataLoader(Data.TensorDataset(X_train, y_train), args.batch_size, shuffle=True)
        self.val_loader = Data.DataLoader(Data.TensorDataset(X_valid, y_valid), args.batch_size, shuffle=False)
        self.test_loader = Data.DataLoader(Data.TensorDataset(X_test, y_test), args.batch_size, shuffle=False)
        pass

    def get_max_paramters(self):
        list_upper = [3,[0],args.layer_top,[1,1],[0,1],args.layer_top,[1,1],[0,1,1],args.layer_top,[1,1]]
        model = Ynet(list_upper,args.feature_number,args)
        model_parameter_max = sum(p.numel() for p in model.parameters() if p.requires_grad and p.dim() > 1)
        args.model_parameter_max = model_parameter_max

