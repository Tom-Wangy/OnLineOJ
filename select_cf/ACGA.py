import numpy as np
import random
import utils
from lower_level import Lower_level
import matplotlib.pyplot as plt
import os
from pretreatment import Data_loader
from hyperparameter import args
import sys
import json


    

    
class ACGA_Algorithm:
    def __init__(self,number) -> None:
        self.args = args
        data = Data_loader(number,args)
        self.weight = data.weight
        self.data = data
        

        self.pop = np.array([])  # 种群
        self.fitness = np.array([])
        self.new_pop = np.array([]) # 子种群
        self.new_fitness = np.array([])
        self.elite_solutions = np.array([])
        pass

    def init_pop(self):
        
        weight = np.array(self.weight)
        # 每个特征的概率
        weight_ = weight/weight.sum()
        pop = []
        index = 1
        while(index <= self.args.pop_size):
            # 先把个题全部初始化为0
            individual_list = []
            for i in range(int(self.args.feature_number)):
                individual_list.append(0)
            # 初始化特征数量
            selected_num =int(self.args.feature_number/5)
            if selected_num > np.count_nonzero(weight_):
                selected_num = np.count_nonzero(weight_)
            print(self.args.feature_number,len(weight_))
            # 选择特征的索引
            index_ = np.random.choice(np.arange(self.args.feature_number), size=int(selected_num), replace=False,p=(weight_/ weight_.sum()))

            # 改编码
            for i in index_:
                individual_list[i]=1
            # 随机生成二进制编码
            for i in range(int(self.args.feature_number),int(self.args.binary_limit)):
                individual_list.append(random.randint(0,1))
            # 随机生成整数编码
            for i in range(int(self.args.integer_limit)):
                individual_list.append(random.randint(self.args.layer_low, self.args.layer_top))
            pop.append(individual_list)
            index += 1
        self.pop =  np.array(pop)
        pass

    def crossover_and_mutation(self):

        new_pop = []
        for father in self.pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father.copy()  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            crossover_rate = np.random.rand()
            mutation_rate = np.random.rand()
            if crossover_rate < self.args.crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = self.pop[np.random.randint(self.args.pop_size-1)]  # 再种群中选择另一个个体，并将该个体作为母亲

                cross_points_start = np.random.randint(low=0, high=self.args.binary_limit+self.args.integer_limit-1)  # 随机产生交叉的点
                cross_points_end = np.random.randint(low=0, high=self.args.binary_limit+self.args.integer_limit - 1)  # 随机产生交叉的点
                if cross_points_start <= cross_points_end:
                    child[cross_points_start:cross_points_end] = mother[cross_points_start:cross_points_end]  # 孩子得到位于交叉点后的母亲的基因
                else:
                    child[cross_points_end:cross_points_start] = mother[cross_points_end:cross_points_start]

            self.mutation(child,mutation_rate)  # 每个后代有一定的机率发生变异
            
            self.mutation_layer_node(child, mutation_rate)
            
            if crossover_rate < self.args.crossover_rate or mutation_rate < self.args.mutation_rate:
                new_pop.append(child.tolist())
                pass
        self.new_pop = np.array(new_pop)
        pass

    def mutation(self,child, mutation_rate):
        select_index = random.randint(0, int(self.args.pop_size/10)-1)
        best_ind = self.elite_solutions.copy()[select_index]

        if mutation_rate < self.args.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            # 获取权值分布
            temp = []
            for i in range(int(self.args.binary_limit)):
                temp.append(1-self.args.turnover_rate)
            temp = np.array(temp)
            # 获取哪些是相同的
            same_list = child[0:int(self.args.binary_limit)] - best_ind[0:int(self.args.binary_limit)]
            same_list = np.abs(same_list)
            probability_list = np.abs(temp-same_list)  # 相同的特征概率变为0.45，不同的变为0.55
            # 选择变异位点
            for i in range(1):
                mutate_point = np.random.choice(np.arange(len(probability_list)), size=1, replace=False,
                                        p=(probability_list / probability_list.sum()))

                if mutate_point < self.args.binary_limit:
                    if child[mutate_point] == 1:
                        child[mutate_point] = 0
                    else:
                        child[mutate_point] = 1  # 将变异点的二进制为反转

            # 下面是ReliefF权重约束
            SD = 2
            for i in range(int(SD)):
                feature_index = [index for index, value in enumerate(child[0:self.args.feature_number]) if value == 0]
                weight_index = np.array(self.weight[feature_index])
                if self.args.feature_selection == 1:
                    # print(weight)
                    constraint_point = np.random.choice(feature_index, size=1, replace=False,p = (np.exp(weight_index) / np.exp(weight_index).sum()))
                    if child[constraint_point] == 0:
                        child[constraint_point] = 1
            SD = 2
            for i in range(int(SD)):
                feature_index = [index for index, value in enumerate(child[0:self.args.feature_number]) if value != 0]
                weight_index = np.array(self.weight[feature_index])
                if self.args.feature_selection == 1:
                    # p1 = np.exp(-weight_index)
                    constraint_point = np.random.choice(feature_index, size=1, replace=False,p = (np.exp(-weight_index) / np.exp(-weight_index).sum()))
                    child[constraint_point] = 0
    
    def mutation_layer_node(self,child, mutation_rate):
        # 随机选择精英解
        select_index = random.randint(0,len(self.elite_solutions) - 1)
        best_ind = self.elite_solutions.copy()[select_index]
        binary_code = child[self.args.feature_number:self.args.feature_number + self.args.layer_number_code]
        layer_number = utils.decode_binary(binary_code)
        # 变异节点和它对应的数值
        mutate_point = np.random.randint(self.args.binary_limit, self.args.binary_limit + layer_number)  # 随机产生一个实数，代表要变异基因的位置
        original_value = child[mutate_point]
        # 在最优个体上变异节点数值
        best_value = best_ind[mutate_point]
        if original_value < best_value:
            mutate_start = original_value
            mutate_end = best_value
        else:
            mutate_start = best_value
            mutate_end = original_value
        if mutation_rate < self.args.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            # 定向变异
            temp_rate = np.random.rand()
            if temp_rate < self.args.turnover_rate:
                child[mutate_point] = random.randint(mutate_start,mutate_end)
            else:
                child[mutate_point] = random.randint(self.args.layer_low,self.args.layer_top)

    def get_fitness(self,pop):
        # pop,decode_pop = utils.check_pop(pop)
        decode_pop = utils.Decode(pop, self.args)
        fitness = []
        for element in decode_pop:
            print(element)
            fitness.append(Lower_level.optimizer_parameter(element,self.data,self.args)["fitness"])
        return np.array(fitness)

    def get_information(self):
        best_fitness = 0
        for i in range(int(self.args.pop_size/10)):
            best_ind_decoder = utils.Decode(self.pop.copy()[i:i+1], self.args)[0]
            best_ind_result = Lower_level.optimizer_parameter(best_ind_decoder, self.data, self.args)
            best_fitness += best_ind_result["fitness"]

        best_fitness /= int(self.args.pop_size/10)
        best_ind_decoder = utils.Decode(self.pop.copy()[0:1], self.args)[0]
        best_ind_result = Lower_level.optimizer_parameter(best_ind_decoder, self.data, self.args)

        best_feature = best_ind_result["FEATURE"]/self.args.feature_number
        best_node = best_ind_result['model_params'] / self.args.model_parameter_max
        best_val_err = 1-best_ind_result['accuracy_test']
        
        return best_fitness, best_val_err, best_node, best_feature

    def strat(self):
        fitness_list = []
        node_list = []
        feature_list =[]
        val_error_list = []
        obj = []
        self.init_pop()
        # 获取适应度
        self.fitness = self.get_fitness(self.pop)
        for i in range(self.args.n_generation):  # 迭代N代
            print("第i次迭代:",i)
            # 获取最优个体
            self.elite_solutions = utils.get_elite(self.pop,self.fitness,self.args)
            # 交叉变异
            self.crossover_and_mutation()
            self.new_fitness = self.get_fitness(self.new_pop)

            self.pop = self.new_pop.tolist() + self.pop.tolist()
            self.fitness = self.new_fitness.tolist() + self.fitness.tolist()
            self.fitness,self.pop = utils.select(np.array(self.pop), np.array(self.fitness),self.args)  # 选择生成新的种群

            best_fitness, best_val_rate, best_node,best_feature = self.get_information()
            fitness_list.append(best_fitness)
            node_list.append(best_node)
            feature_list.append(best_feature)
            val_error_list.append(best_val_rate)
        print(self.pop)
        print(self.fitness)

        # train_result = ensemble_learning(pop,args)
        # accuracy = print_info(pop)
        # plt.xlabel('The number of generations')
        # plt.ylabel('Fitness')
        # plt.plot(range(int(self.args.n_generation)), fitness_list)
        # plt.savefig(result_picture + '/' +  jpg_head + '.png', dpi=120)
        # plt.show()
        # plt.close()


        # return  train_result

    @staticmethod
    def Acga_Run(number):

        # 首先打开文件描述符
        stdout_fd = os.open('/root/C++_program/OnlineOJ/select_cf/tmp/stdout.txt', os.O_WRONLY | os.O_CREAT)
        stderr_fd = os.open('/root/C++_program/OnlineOJ/select_cf/tmp/stderr.txt', os.O_WRONLY | os.O_CREAT)

        # if(stdout_fd < 0 or stderr_fd < 0):

        #     return -1
        
        # 创建子进程，所有的事情交给子进程来处理
        pid = os.fork()

        if pid == 0:
            # 子进程

            # 重定向标准输出和标准错误到父进程打开的文件描述符
            os.dup2(stdout_fd, 1)  # 将文件描述符stdout_fd复制到文件描述符1，即标准输出
            os.dup2(stderr_fd, 2)  # 将文件描述符stderr_fd复制到文件描述符2，即标准错误


            # 使用程序替换执行新的程序
            obj = ACGA_Algorithm(number)

            obj.strat()

            # 关闭文件描述符
            os.close(stdout_fd)
            os.close(stderr_fd)

        else:
            # 父进程

            # 关闭文件描述符
            os.close(stdout_fd)
            os.close(stderr_fd)

            # 等待子进程结束
            _, status = os.waitpid(pid, 0)

            # 创建用于输出的json
            output_data = {
                'std_out': None,
                'std_err':None
            }

            # 在父进程中读取子进程输出的文件内容
            with open('/root/C++_program/OnlineOJ/select_cf/tmp/stdout.txt', 'r') as stdout_file:
                output_data['std_out'] = stdout_file.read()
                pass

            with open('/root/C++_program/OnlineOJ/select_cf/tmp/stderr.txt', 'r') as stderr_file:
                output_data['std_err'] = stderr_file.read()
                pass
            # output_data = json.dumps(output_data)

            os.unlink('/root/C++_program/OnlineOJ/select_cf/tmp/stdout.txt')
            os.unlink('/root/C++_program/OnlineOJ/select_cf/tmp/stderr.txt')
        return output_data


        