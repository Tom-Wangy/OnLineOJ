import torch
import numpy as np
from hyperparameter import args
import random

def decode_binary(list_binary):
    '''
    :param list_binary: binary coded list
    :return:decimal number
    '''
    list_binary = list_binary.tolist()
    std = [1,2,4,8,16,32,64,128]
    i = 0
    dec = 0
    while list_binary != []:
        temp = list_binary.pop()
        dec = temp * std[i] + dec
        i += 1
    if dec == 0:
        return 1
    else:
        return dec
def Inite_Pop1(weight,args):
    '''
    :param pop_size: 种群数量
    :param weight: 归一化后的特征权值
    :param args: 超参数
    :return: 返回结果是ndarray形式的种群
    '''
    weight = np.array(weight)
    # 每个特征的概率
    weight_ = weight/weight.sum()
    print("weight",weight.tolist())
    pop = []
    index = 1
    while(index <= args.pop_size):
        # 先把个题全部初始化为0
        individual_list = []
        for i in range(int(args.feature_number)):
            individual_list.append(0)
        # 初始化特征数量
        selected_num = random.randint(int(args.feature_number/6),int(args.feature_number/3))
        sorted_lst = sorted(enumerate(weight), key=lambda x: x[1], reverse=True)
        sorted_index = [index for index, _ in sorted_lst]
        sorted_values = np.array([value for _, value in sorted_lst])

        selected_list = []
        weight1=sorted_values[0:int(args.feature_number/3)]
        weight_1 = weight1/weight1.sum()
        selected_list1 = np.random.choice(sorted_index[0:int(args.feature_number/3)], size=4*int(selected_num/6), replace=False,
                                 p=(np.exp(weight_1) / np.exp(weight_1).sum()))
        selected_list += selected_list1.tolist()
        weight2 = sorted_values[int(args.feature_number / 3):2*int(args.feature_number / 3)]
        weight_2 = weight2 / weight2.sum()
        selected_list2 = np.random.choice(sorted_index[int(args.feature_number / 3):2*int(args.feature_number / 3)], size=1*int(selected_num/6),
                                          replace=False,
                                          p=(np.exp(weight_2) / np.exp(weight_2).sum()))
        selected_list += selected_list2.tolist()
        weight3 = sorted_values[2*int(args.feature_number / 3):int(args.feature_number)]
        weight_3 = weight3 / weight3.sum()
        selected_list3 = np.random.choice(sorted_index[2*int(args.feature_number / 3):int(args.feature_number)],
                                          size=int(selected_num/6),
                                          replace=False,
                                          p=(np.exp(weight_3) / np.exp(weight_3).sum()))
        selected_list += selected_list3.tolist()

        # 改编码
        for i in selected_list:
            # rd = random.uniform(0,1)
            # if(rd < weight[i]):
            individual_list[i]=1
        # 随机生成二进制编码
        for i in range(int(args.feature_number),int(args.binary_limit)):
            individual_list.append(random.randint(0,1))
        # 随机生成整数编码
        for i in range(int(args.integer_limit)):
            individual_list.append(random.randint(args.layer_low, args.layer_top))
        pop.append(individual_list)
        index += 1
    return np.array(pop)
def Inite_Pop(weight,args):
    '''
    :param pop_size: 种群数量
    :param weight: 归一化后的特征权值
    :param args: 超参数
    :return: 返回结果是ndarray形式的种群
    '''
    weight = np.array(weight)
    # 每个特征的概率
    weight_ = weight/weight.sum()
    print("weight",weight.tolist())
    pop = []
    index = 1
    while(index <= args.pop_size):
        # 先把个题全部初始化为0
        individual_list = []
        for i in range(int(args.feature_number)):
            individual_list.append(0)
        # 初始化特征数量
        selected_num =int(args.feature_number/5)
        if selected_num > np.count_nonzero(weight_):
            selected_num = np.count_nonzero(weight_)
        print(args.feature_number,len(weight_))
        # 选择特征的索引

        # indices = np.argpartition(weight, -int(args.feature_number/2))[-int(args.feature_number/2):]
        # sorted_indices = indices[np.argsort(weight[indices])[::-1]]
        # sorted_values = weight[sorted_indices]
        index_ = np.random.choice(np.arange(args.feature_number), size=int(selected_num), replace=False,
                                 p=(weight_/ weight_.sum()))

        # index_ = random.sample(range(len(individual_list)), selected_num)

        # 改编码
        for i in index_:
            # rd = random.uniform(0,1)
            # if(rd < weight[i]):
            individual_list[i]=1
        # 随机生成二进制编码
        for i in range(int(args.feature_number),int(args.binary_limit)):
            individual_list.append(random.randint(0,1))
        # 随机生成整数编码
        for i in range(int(args.integer_limit)):
            individual_list.append(random.randint(args.layer_low, args.layer_top))
        pop.append(individual_list)
        index += 1
    return np.array(pop)

def max_individual(args):
    ind = []
    # fLag = 0 表示网络层数上界大于特征数量上界
    if(args.feature_number > args.layer_top):
        flag = 1
    else:
        flag = 0
    #使用所有特征
    for i in range(int(args.feature_number)):
        ind.append(1)
    for i in range(int(args.feature_number), int(args.binary_limit)):
        ind.append(1)

    for i in range(int(args.integer_limit)):
        if (i % 2 == 0):
            if (i == 0):
                ind.append(0)
                continue
            elif(i == 1):
                ind.append(1)
            else:
                if(args.feature_number > args.layer_top):
                    ind.append(1)
                else:
                    ind.append(i-1)
        else:
            ind.append(random.randint(args.layer_top))
    return ind

def get_elite(pop, fitness, args):
    index = np.argsort(fitness)
    index = index.tolist()

    pop_index = pop[index]
    fitness_index = fitness[index]
    best_pop = []
    best_decode_pop = []
    for i in range(len(pop_index)):
        # 解码
        temp_pop = Decode(pop_index[i:i + 1], args)
        if len(best_decode_pop) < int(args.pop_size/10):
            if temp_pop[0] not in best_decode_pop:
                best_decode_pop.append(temp_pop[0])
                best_pop.append(pop_index.copy()[i].tolist())
        else:
            break
    return np.array(best_pop)

def crossover_and_mutation(pop, args, best_ind, weight):
    '''
    :param pop: pop
    :param args: model hyperparameter
    :param best_ind:elite individuals of pop
    :param weight:feature weight
    :return:current population progeny
    '''
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father.copy()  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        crossover_rate = np.random.rand()
        mutation_rate = np.random.rand()
        if crossover_rate < args.crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(args.pop_size-1)]  # 再种群中选择另一个个体，并将该个体作为母亲

            cross_points_start = np.random.randint(low=0, high=args.binary_limit+args.integer_limit-1)  # 随机产生交叉的点
            cross_points_end = np.random.randint(low=0, high=args.binary_limit+args.integer_limit - 1)  # 随机产生交叉的点
            if cross_points_start <= cross_points_end:
                child[cross_points_start:cross_points_end] = mother[cross_points_start:cross_points_end]  # 孩子得到位于交叉点后的母亲的基因
            else:
                child[cross_points_end:cross_points_start] = mother[cross_points_end:cross_points_start]

        mutation(child,mutation_rate,best_ind,weight,args)  # 每个后代有一定的机率发生变异
        # mutation(child, mutation_rate,best_ind,weight,args)  # 每个后代有一定的机率发生变异
        mutation_layer_node(child, mutation_rate,best_ind,args)
        # mutation_layer_node(child, mutation_rate,best_ind,args)
        if crossover_rate < args.crossover_rate or mutation_rate < args.mutation_rate:
            new_pop.append(child.tolist())

    return np.array(new_pop)

def select(pop, fitness,args):  # nature selection wrt pop's fitness
    index = np.argsort(fitness)
    index = index.tolist()
    # 降序排列
    pop_descend = pop[index]
    fitness_descend = fitness[index]

    elites_fitness = []  # 精英解的适应度
    elites_pop = []  # 精英解
    elites_decode_pop = []  # 解码后的精英解
    elites_index = []  # 精英解的索引
    for i in range(len(pop_descend)):
        # 解码
        temp_pop = Decode(pop_descend[i:i+1],args)
        if len(elites_decode_pop) < int(args.pop_size/10):
            # print(temp_pop[0])
            # print(elites_decode_pop)
            if temp_pop[0] not in elites_decode_pop :
                elites_decode_pop.append(temp_pop[0])
                elites_pop.append(pop_descend.copy()[i].tolist())
                elites_index.append(i)
                elites_fitness.append(fitness_descend.copy()[i].tolist())
        else:
            break


    pop_none_elite = np.delete(pop_descend.copy(),elites_index,axis=0)
    fitness_none_elite = np.delete(fitness_descend.copy(),elites_index,axis=0)

    # p1 = ((1-fitness_none_elite)/(1-fitness_none_elite).sum())
    # index = np.random.choice(np.arange(len(fitness_none_elite)),size=args.pop_size - int(args.pop_size/10), replace=False ,p=((1-fitness_none_elite)/(1-fitness_none_elite).sum()))
    # p2 = p=(np.exp(-fitness_none_elite) / np.exp(-fitness_none_elite).sum())
    index = np.random.choice(np.arange(len(fitness_none_elite)), size=args.pop_size - int(args.pop_size/10), replace=False,
                                        p=(np.exp(-fitness_none_elite) / np.exp(-fitness_none_elite).sum()))

    pop = pop_none_elite.copy()[index]  # 选择出来的非精英解
    fitness = fitness_none_elite.copy()[index]  # 选择出来的非精英解的适应度

    pop = pop.tolist()
    pop = elites_pop + pop
    fitness = fitness.tolist()
    fitness = elites_fitness + fitness


    return np.array(fitness), np.array(pop)  # fitness,pop

def mutation(child, mutation_rate,elite_solutions,weight,args):
    '''
    :param child: child
    :param mutation_rate: mutation rate
    :param best_ind:current generation elite individuals
    :param weight:feature weight
    :param args:model hyperparameter
    :return:Postmutated individual
    '''
    # 随机选择一个精英解
    select_index = random.randint(0, int(args.pop_size/10)-1)
    best_ind = elite_solutions.copy()[select_index]

    if mutation_rate < args.mutation_rate:  # 以MUTATION_RATE的概率进行变异
        # 获取权值分布
        temp = []
        for i in range(int(args.binary_limit)):
            temp.append(1-args.turnover_rate)
        temp = np.array(temp)
        # 获取哪些是相同的
        same_list = child[0:int(args.binary_limit)] - best_ind[0:int(args.binary_limit)]
        same_list = np.abs(same_list)
        probability_list = np.abs(temp-same_list)  # 相同的特征概率变为0.45，不同的变为0.55
        # 选择变异位点
        for i in range(1):
            mutate_point = np.random.choice(np.arange(len(probability_list)), size=1, replace=False,
                                     p=(probability_list / probability_list.sum()))

            if mutate_point < args.binary_limit:
                if child[mutate_point] == 1:
                    child[mutate_point] = 0
                else:
                    child[mutate_point] = 1  # 将变异点的二进制为反转

        # 下面是ReliefF权重约束
        SD = 2#sum(child[0:args.feature_number]) / 40
        for i in range(int(SD)):
            feature_index = [index for index, value in enumerate(child[0:args.feature_number]) if value == 0]
            weight_index = np.array(weight[feature_index])
            if args.feature_selection == 1:
                # print(weight)
                constraint_point = np.random.choice(feature_index, size=1, replace=False,p = (np.exp(weight_index) / np.exp(weight_index).sum()))
                if child[constraint_point] == 0:
                    child[constraint_point] = 1
        SD = 2 #sum(child[0:args.feature_number])/20

        for i in range(int(SD)):
            feature_index = [index for index, value in enumerate(child[0:args.feature_number]) if value != 0]
            weight_index = np.array(weight[feature_index])
            if args.feature_selection == 1:
                # p1 = np.exp(-weight_index)
                constraint_point = np.random.choice(feature_index, size=1, replace=False,p = (np.exp(-weight_index) / np.exp(-weight_index).sum()))
                child[constraint_point] = 0


def mutation_layer_node(child, mutation_rate, elite_solutions,args):
    # 随机选择精英解
    select_index = random.randint(0,len(elite_solutions) - 1)
    best_ind = elite_solutions.copy()[select_index]
    binary_code = child[args.feature_number:args.feature_number + args.layer_number_code]
    layer_number = decode_binary(binary_code)
    # 变异节点和它对应的数值
    mutate_point = np.random.randint(args.binary_limit, args.binary_limit + layer_number)  # 随机产生一个实数，代表要变异基因的位置
    original_value = child[mutate_point]
    # 在最优个体上变异节点数值
    best_value = best_ind[mutate_point]
    if original_value < best_value:
        mutate_start = original_value
        mutate_end = best_value
    else:
        mutate_start = best_value
        mutate_end = original_value
    if mutation_rate < args.mutation_rate:  # 以MUTATION_RATE的概率进行变异
        # 定向变异
        temp_rate = np.random.rand()
        if temp_rate < args.turnover_rate:
            child[mutate_point] = random.randint(mutate_start,mutate_end)
        else:

            child[mutate_point] = random.randint(args.layer_low,args.layer_top)


def Decode(pop,args):
    '''
    :param pop:population
    :return:decoder population
    '''
    decode_pop = []
    for element in pop:
        decode_individual = []
        # 第一部分是选取的特征数量及其对应的指标
        if args.feature_selection == 0:
            feature_number = 0
        else:
            feature_number = args.feature_number
            decode_individual.append(sum(element[0:feature_number]))
            for i in range(feature_number):
                if element[i] == 1:
                    decode_individual.append(i)

        flag = feature_number  # 解码到flag这个位置了
        # 获得网络层数和每层跳转连接数、节点数、激活函数
        layer_code = element[flag:flag+args.layer_number_code]  # 网络层数对应的二进制编码
        flag += args.layer_number_code  # 更新解码到的位置
        layer_decode = decode_binary(layer_code)  # 网络层数的解码
        decode_individual.append(layer_decode)  # 添加到解码个体中


        for i in range(int(layer_decode)):
            # if i == 0:
            if(i==0):
                decode_individual.append([])
            else:
                connect_code = element[flag:flag+i]  # 表示连接的编码
                flag += i  # 更新标志位
                decode_individual.append(connect_code.tolist())  # 添加连接隐层
            decode_individual.append(element[int(args.binary_limit+i)])  # 添加隐层节点数
            activation_code = element[flag:flag + 2]  # 激活函数编码
            flag += 2
            decode_individual.append(activation_code.tolist())  # 添加激活函数编码
            # else:
            #     decode_individual.append(element[int(args.binary_limit+2*i)])  # 添加连接层
            #     decode_individual.append(element[int(args.binary_limit+2*i+1)])  # 添加隐层节点数
            #     activation_code = element[flag:flag+2]
            #     decode_individual.append(activation_code)  # 添加激活函数
            #     flag = flag+2
        decode_pop.append(decode_individual)
    return decode_pop

def round_pred(pred):
    return torch.tensor(np.argmax(pred,axis=1))



def softmax(x):
    """ softmax function """
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x



def get_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
