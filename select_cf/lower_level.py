import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils import round_pred
# from utils import get_info,Decode,get_normalization
import  utils
import matplotlib.pyplot as plt
import time
from skrebate import ReliefF
import torch
import torch.nn as nn
from model import Ynet
import multiprocessing
from torch.autograd import Variable
import warnings

# 捕获和处理警告
warnings.filterwarnings("ignore") 

def child_process(LIST,train_loader,val_loader,test_loader,args,data):
    # 处理特征
    if args.feature_selection == 1:
        FEATURE = LIST.pop(0)
        FEATURE_INDEX = []
        for i in range(FEATURE):
            FEATURE_INDEX.append(LIST.pop(0))
    else:
        FEATURE = args.feature_number
        FEATURE_INDEX = [i for i in range(FEATURE)]
        

    model = Ynet(LIST,FEATURE,args)

    # loss
    criterion = nn.CrossEntropyLoss(size_average=False)  # B
    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    test_losses = []
    best_acc = 0

    torch_val_best, torch_val_y_best = torch.tensor([]), torch.tensor([])

    for epoch in range(args.epoch):
        repres_list, label_list = [], []

        torch_train, torch_train_y = torch.tensor([]), torch.tensor([])
        torch_val, torch_val_y,torch_test,torch_test_y,torch_class = torch.tensor([]), torch.tensor([]),torch.tensor([]), torch.tensor([]),torch.tensor([])

        model.train()  # !!! Train
        correct = 0
        train_loss = 0
        for step, (train_x, train_y) in enumerate(train_loader):

            train_x = Variable(train_x, requires_grad=False)
            train_y = Variable(train_y, requires_grad=False)
            train_x = train_x[:,FEATURE_INDEX]
            predict_y = model(train_x)
            loss = criterion(predict_y, train_y)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 反向传播

            repres_list.extend(predict_y.cpu().detach().numpy())
            label_list.extend(train_y.cpu().detach().numpy())

            pred = utils.round_pred(predict_y.data.cpu().numpy())  # B
            correct += pred.eq(train_y.view_as(pred)).sum()

            train_loss += loss.item() * len(train_y)

        # 计算本次迭代平均损失
        train_losses.append(train_loss / len(train_y))
        # 计算精度
        accuracy_train = 100. * correct / len(train_y)
        

        model.eval()  # !!! Valid
        val_loss = 0
        correct = 0
        repres_list_valid, label_list_valid = [], []

        with torch.no_grad():
            for step, (valid_x, valid_y) in enumerate(val_loader):  # val_loader
                valid_x = Variable(valid_x, requires_grad=False)
                valid_x = valid_x[:,FEATURE_INDEX]
                valid_y = Variable(valid_y, requires_grad=False)

                optimizer.zero_grad()  # --->>
                y_hat_val = model(valid_x)

                loss = criterion(y_hat_val, valid_y.type(torch.LongTensor))  # B
                val_loss += loss * len(valid_y)  # sum up batch loss

                label_list_valid.extend(valid_y.detach().numpy().tolist())

                pred_val = round_pred(y_hat_val.data.cpu().numpy())  # B

                correct += pred_val.eq(valid_y.view_as(pred_val)).sum()
                pred_prob_val = y_hat_val  # B
                torch_val = torch.cat([torch_val, pred_prob_val.data.cpu()], dim=0)
                torch_val_y = torch.cat([torch_val_y, valid_y.data.cpu()], dim=0)
        val_losses.append(val_loss / len(valid_y))  # all loss / all sample
        accuracy_valid = correct / len(valid_y)

        test_loss = 0
        correct = 0

        repres_list_test, label_list_test = [], []
        accuracy_test = 0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(test_loader):  # val_loader
                test_x = Variable(test_x, requires_grad=False)
                test_x = test_x[:,FEATURE_INDEX]
                test_y = Variable(test_y, requires_grad=False)

                optimizer.zero_grad()  # --->>
                y_hat_test = model(test_x)

                loss = criterion(y_hat_test, test_y.type(torch.LongTensor))  # B
                test_loss += loss * len(test_y)  # sum up batch loss

                
                label_list_test.extend(test_y.cpu().detach().numpy().tolist())

                pred_test = round_pred(y_hat_test.data.cpu().numpy())  # B

                # get the index of the max log-probability
                correct += pred_test.eq(test_y.view_as(pred_test)).sum().item()
 				#C
                pred_prob_test = y_hat_test  # B
                torch_test = torch.cat([torch_test, pred_prob_test.data.cpu()], dim=0)
                # torch_class = torch.cat([torch_class, pred_test], dim=0)
                torch_test_y = torch.cat([torch_test_y, test_y.data.cpu()], dim=0)
        test_losses.append(test_loss / len(test_y))  # all loss / all sample
        accuracy_test = correct / len(test_y)



    print('accuracy_valid:%f'% (accuracy_valid))
    

    # 模型参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.dim() > 1)
    param_raio = model_params/args.model_parameter_max

    fun_fitness = 0.98*(1-accuracy_valid) + 0.01 * param_raio + 0.01*FEATURE/args.feature_number

    data['fitness'] = fun_fitness
    data['accuracy_valid'] = accuracy_valid
    data['model_params'] = model_params
    data['FEATURE'] = FEATURE
    data['accuracy_test'] = accuracy_test
    data['torch_test'] = torch_test




    pass

class Lower_level:

    def __init__(self) -> None:
        # 不需要任何参数
        pass

    @staticmethod
    def optimizer_parameter(LIST,data,args):
        '''
        LIST:上层传递给下层的特征与网络结构
        args:文章中的所有超参数
        return 训练好的参数，选择的特征
        '''

        #创建两个进程，父进程用于检查是否形成想要的结果，子进程运行程序
        #这么做防止程序出错，导致整个进程崩溃
        manager = multiprocessing.Manager()
        data_dict = manager.dict()

        process = multiprocessing.Process(target=child_process, args=(LIST,data.train_loader,data.val_loader,data.test_loader,args,data_dict))
        process.start()  # 创建并运行子进程
        process.join()  # 等待子进程
        

        result_dict = {}
        result_dict.update(data_dict)
        
        return data_dict
