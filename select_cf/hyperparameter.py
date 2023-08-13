# import argparse

# parser = argparse.ArgumentParser(description='PyTorch bi-level programmingModel')

# # Feature selection hyperparameter
# parser.add_argument('--feature_selection', type=int, default=1,
#                     help='decide whether to do feature selection')

# # Data-dependent hyperparameters
# parser.add_argument('--dataset', type=str, default='sonar',
#                     help='location of the data corpus')
# parser.add_argument('--data', type=str, default='./data/'+ parser.parse_args().dataset + '.txt',
#                     help='location of the data corpus')
# parser.add_argument('--feature_number', type=int, default = 60,
#                     help='number of features in the sample')
# parser.add_argument('--class_number', type=int, default= 2,
#                     help='number of categories')

# # Network hyperparameter
# parser.add_argument('--layer_number', type=int, default=3,
#                     help='upper bound of the network layer,It is always a summation of primes of 2, like 1, 3, 7')
# parser.add_argument('--layer_low', type=int, default=10,
#                     help='lower bound for the number of nodes at each layer')
# parser.add_argument('--layer_top', type=int, default=100,#1 * parser.parse_args().feature_number,
#                     help='upper bound for the number of nodes at each layer')
# parser.add_argument('--epoch', type=int, default=20,
#                     help='number of network iterations')
# parser.add_argument('--batch_size', type=int, default=500,
#                     help='batch_size')
# parser.add_argument('--activation', type=int, default=2,
#                     help='number of active functions')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='learning rate')

# # Evolutionary algorithm hyperparameter
# parser.add_argument('--pop_size', type=int, default=30,
#                     help='population number')
# parser.add_argument('--crossover_rate', type=float, default=0.5,
#                     help='crossover rate')
# parser.add_argument('--mutation_rate', type=float, default=0.8,
#                     help='mutation rate')
# parser.add_argument('--turnover_rate', type=float, default=0.55,
#                     help='asymmetric turnover rate')
# parser.add_argument('--n_generation', type=int, default=10,
#                     help='upper bound on the number of iterations of population evolution')
# parser.add_argument('--model_parameter_max', type=int, default=1000000,help='model paramter max')
# parser.add_argument('--SD', type=int, default= 1,help='基于reliefF的变异算子超参数')
# parser.add_argument('--SM', type=int, default= 3,help='学习机制变异位点个数')
# parser.add_argument('--generation_txt', type=str, default='./result/error.txt',help='保留每一代结果')
# parser.add_argument('--duplicate_experimental_path', type=str, default='./result/error.txt',help='单独运行一次结果')
# args = parser.parse_args()

# def get_code_hyperparameter(parser):
#     '''
#     :param args: 算法超参数
#     :return: 编码超参数
#     '''
#     N3 = 2  # 网络层数占2位
#     N4 = parser.parse_args().layer_number * (0+parser.parse_args().layer_number-1) / 2 # 用于残差连接 二进制编码
#     N5 = 2 * parser.parse_args().layer_number  # 用于激活函数
#     N6 = parser.parse_args().layer_number  # 整数编码用于每层节点数
#     if parser.parse_args().feature_selection == 1:
#         BIN_LIMIT = parser.parse_args().feature_number + N3 + N4 + N5  # 二进制编码和整数编码分解点或者叫二进制编码个数
#     else:
#         BIN_LIMIT = N3 + N4 + N5  # 二进制编码和整数编码分解点或者叫二进制编码个数
#     INTEGER_LIMIT = parser.parse_args().layer_number  # 整数编码个数
#     parser.add_argument('--binary_limit', type=int, default=BIN_LIMIT,
#                     help='number of binary codes')
#     parser.add_argument('--integer_limit', type=int, default=INTEGER_LIMIT,
#                       help='number of integer codes')
#     parser.add_argument('--layer_number_code', type=int, default=N3,
#                         help='The coding length of the network layer')
# get_code_hyperparameter(parser)
# args = parser.parse_args()


class Args:
    def __init__(self) -> None:
        self.feature_selection=1
        self.dataset = 'sonar'
        self.data = './data/'+ self.dataset + '.txt'
        self.feature_number = 60
        self.class_number = 2

        self.layer_number = 3
        self.layer_low=10
        self.layer_top=100
        self.epoch=20
        self.batch_size=500
        self.activation=2
        self.lr=0.01

        self.pop_size=30
        self.crossover_rate=0.5
        self.mutation_rate=0.8
        self.turnover_rate=0.55
        self.n_generation=3
        self.model_parameter_max=1000000
        self.SD= 1
        self.SM= 3
        self.generation_txt='./result/error.txt'
        self.duplicate_experimental_path='./result/error.txt'
        self.get_other()

    def get_other(self):
    
        N3 = 2  # 网络层数占2位
        N4 = self.layer_number * (0+self.layer_number-1) / 2 # 用于残差连接 二进制编码
        N5 = 2 * self.layer_number  # 用于激活函数
        N6 = self.layer_number  # 整数编码用于每层节点数
        if self.feature_selection == 1:
            BIN_LIMIT = self.feature_number + N3 + N4 + N5  # 二进制编码和整数编码分解点或者叫二进制编码个数
        else:
            BIN_LIMIT = N3 + N4 + N5  # 二进制编码和整数编码分解点或者叫二进制编码个数
        INTEGER_LIMIT = self.layer_number  # 整数编码个数
        self.binary_limit=BIN_LIMIT
        self.integer_limit=INTEGER_LIMIT
        self.layer_number_code=N3
        pass
    pass

args = Args()


