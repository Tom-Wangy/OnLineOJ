import torch
import torch.nn as nn

class Ynet(nn.Module):
    def __init__(self,LIST,FEATURE,args):
        '''
        :param LIST: 结构编码
        :param FEATURE: 选择特征的数量
        :param args: 超参数
        '''
        super(Ynet, self).__init__()

        self.LIST = LIST
        self.feature_number = FEATURE

        # 创建列表保存每一层的权值与激活函数
        self.list = self.get_initlayers()
        # 分类层
        self.classifier1 = nn.Linear(self.LIST[-2], args.class_number)


    # ---------------->>>
    def forward(self, x):
        x = self.get_layer(x)
        x = self.classifier1(x)

        return x

    def get_layer(self,x):
        # 用于保存每一层结果
        list_temp = []
        list_temp.append(x)
        # 网络层数
        layer_num = self.LIST[0]
        # 获取每一层结构并放入self.list方便后面调用
        for i in range(layer_num):
            # 第1层只能连接上一层
            if i == 0:
                x = self.list[i](x)
                list_temp.append(x)
            else:
                # 输入节点
                connect_code = self.LIST[3*i+1]  #连接编码
                for j in range(len(connect_code)): # 循环拼接

                    if connect_code[j] == 1:
                        x = torch.hstack((list_temp[j], x))

                x = self.list[i](x)  # 计算第i+1层输出
                list_temp.append(x)  # 添加结果
                
        return x

    def get_initlayers(self):
        layer_list = nn.ModuleList()
        # 网络层数
        layer_num = self.LIST[0]
        # 获取每一层结构并放入self.list方便后面调用
        for i in range(layer_num):
            # 第1层只能连接上一层
            if i == 0:
                x = nn.Sequential(nn.Linear(int(self.feature_number), int(self.LIST[3 *i + 2])),
                                  self.get_activation(self.LIST[3 *i + 3])
                                  )
                layer_list.append(x)
            else:
                # 连接的之前的节点
                connect_code = self.LIST[3 * i + 1]  # 连接编码
                input_node = self.LIST[3 * (i - 1) + 2]  # 第i+1层的输入节点数量，初始化为i层节点数
                for j in range(len(connect_code)):
                    if j == 0:  # 判断第一位
                        input_node += self.feature_number if connect_code[j] == 1 else  0
                    else:
                        if connect_code[j] == 1:
                            input_node += self.LIST[(j - 1)*3+2]
                # 构建
                x = nn.Sequential(nn.Linear(int(input_node), int(self.LIST[3 * i + 2])),
                                  self.get_activation(self.LIST[3 * i + 3]))
                layer_list.append(x)

        return layer_list


    def get_classifer(self,CLASS_NUMBER,x):
        '''
        :param CLASS_NUMBER: 类别
        :param x:
        :return: 分类模块
        '''

        self.classifier = nn.Linear(x.size(1),CLASS_NUMBER)

        return self.classifier(x)

    # 获取激活函数
    def get_activation(self,activation_coding):
        '''
        :param num:激活函数对应编码
        :return:
        '''
        # 越界访问
        # assert num <= args.activation and num >= 1

        if(activation_coding[0] == 0 and activation_coding[1] == 0):
            return nn.ReLU()
        elif activation_coding[0] == 1 and activation_coding[1] == 0:
            return nn.Identity()
        elif activation_coding[0] == 1 and activation_coding[1] == 0:
            return nn.Sigmoid()
        else:
            return nn.Tanh()