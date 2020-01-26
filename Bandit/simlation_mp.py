import csv
import time
import os
import sys
import multiprocessing
from collections import Counter
from multiprocessing import Pool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import environment as en
import agent as ag
import simlations


def switch_calcu(sikou_list):
    switch_array = []
    pre_selected_arm = 1000
    for sikou in sikou_list:
        for selected in sikou:

            x = selected == pre_selected_arm
            if x:
                add = 0
            else:
                add = 1
            pre_selected_arm = selected
            switch_array.append(add)


def make_dir(dir_name):
    file_path = os.path.dirname(os.getcwd() + '/output' + dir_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def multi_process(env, sampleList):
    p = Pool(multiprocessing.cpu_count())
    # print(sampleList)
    output = p.map(env.sim, sampleList)
    p.close()
    return output


def args_parser(args):
    is_nonstationary = int(args[1])
    is_constant = int(args[2])
    sim_name = args[3]
    switch = int(args[4])
    sim_num = int(args[5])
    print("is_nos:"+str(is_nonstationary))
    print("is_cons:"+str(is_constant))
    print("sim_name:"+str(sim_name))
    print('sim_num:'+str(sim_num))
    print('switch:' + str(switch))

    if is_nonstationary == 1:
        str1 = 'nonstationary_'
    else:
        str1 = 'stationary_'
    if is_constant == 1:
        str2 = 'constant_'
    else:
        str2 = 'nonconstant_'
    if print_flag:
        print('will output csv')
    else:
        print('will not output csv')
    return is_nonstationary, is_constant, sim_name, switch, sim_num, str1, str2

if __name__ == '__main__':
    args = sys.argv
    multiprocess = True
    print_flag = True
    is_nonstationary, is_constant, sim_name, switch, sim_num, str1, str2 = args_parser(args)
    
    env = simlations.Simlations(sim_name)

    dir_csv_name = "csv/"
    dir_switch = "switch"+str(switch)+"/"
    path = "/"+dir_switch+dir_csv_name+sim_name+'/'
    make_dir(path)
    agent_regrets = {}

    # 時間計測
    start = time.time()

    # 実行するアルゴリズムのリスト
    args_list = ["UCB1T", "TS"]

    if multiprocess:
        print('multi')
        for x in range(len(args_list)):
            arg_list = [(args_list[x], 100000, 20, is_nonstationary, is_constant, switch, i) for i in range(sim_num)]
            # alg, step_num, k, is_nonstationary, is_constant, switch, seed
            output = multi_process(env, arg_list)
            # switch dousa: heikien toranai
            # regrets = np.mean(output, 0)
            regrets = output
            switch_calcu(regrets)
            agent_regrets.update({args_list[x]: regrets})
            # print(output)
    else:
        print('no_ multi')
        for x in range(len(args_list)):
            arg_list = [(args_list[x], 100000, 20, is_nonstationary, is_constant, switch, i) for i in range(sim_num)]
            # alg, step_num, k, is_nonstationary, is_constant, switch, seed
            output = [env.sim(arg_list[i]) for i in range(sim_num)]
            regrets = np.mean(output, 0)
            agent_regrets.update({args_list[x]: regrets})

    df = pd.DataFrame(agent_regrets)
    if print_flag:
        if str(sim_name) == 'sim_switchtest':
            print("output to csv:", path + str1 + str2 + 'switches.csv')
            df.to_csv('.' + path + str1 + str2 + 'switches.csv')
        else:
            print("output to csv:", path+str1+str2+'regrets.csv')
            df.to_csv('.'+path+str1+str2+'regrets.csv')
    else:
        print('no output csv')

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
