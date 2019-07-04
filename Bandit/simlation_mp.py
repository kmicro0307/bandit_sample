import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag
import multiprocessing
from collections import Counter
from multiprocessing import Pool
import pandas as pd
import csv
import time
import os
import sys
import time

def make_dir(dir_name):
    file_path = os.path.dirname(os.getcwd() + dir_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
def simulation(alg, step_num, k, is_nonstationary, is_constant, switch, seed):
    np.random.seed(seed=seed)
    labels = ["UCB1T", "TS", "RS", "RS OPT", "TS gamma", "RS gamma", "RS OPT gamma", "meta UCB1T", "meta TS", "meta RS",
              "meta RS OPT"]
    colors = ["g", "b", "r", "m", "c", "#ff7f00", "#a65628", "g", "b", "r", "m"]
    accuracy = np.zeros(step_num)
    regrets = np.zeros(step_num)

    # for sim in range(simulation_num):
    # print(sim + 1)
    bandit = en.Bandit(k)
    agent_list = {"UCB1T": ag.UCB1T(k),
                  "TS": ag.TS(k),
                  "RS_gamma": ag.RS_gamma(k, gamma=1.0),
                  "RS_OPT": ag.RS_OPT(k),
                  "TS_gamma": ag.TS_gamma(k),
                  "RS_gamma2": ag.RS_gamma(k),
                  "RS_OPT_gamma": ag.RS_OPT_gamma(k),
                  "meta_UCB1T": ag.meta_bandit(k, agent=ag.UCB1T(k), higher_agent=ag.UCB1T(2),
                                              l=500, delta=0, lmd=30),
                  "meta_TS": ag.meta_bandit(k, agent=ag.TS(k),
                                            higher_agent=ag.TS(2), l=30, delta=0, lmd=30),
                  "meta_RS_gamma": ag.meta_bandit(k, agent=ag.RS_gamma(k, gamma=1.0), higher_agent=
                  ag.RS_gamma(2, gamma=1.0), l=30, delta=0, lmd=30),
                  "meta_RS_OPT": ag.meta_bandit(k, agent=ag.RS_OPT(k), higher_agent=ag.RS_OPT(2), l=30,
                                                delta=0, lmd=30)}

    agent = agent_list[alg]
    agent.opt_r = bandit.opt_r
    regret_sums = 0
    # prev_selecteds = np.zeros(agent_num)

    for step in range(step_num):
        prev_selected = 0
        regret = 0

        # バンディット変化
        if is_nonstationary:
            if is_constant:
                if step % 10000 == 0:
                    bandit = en.Bandit(k)
                    # for agent in agent_list:
                    agent.opt_r = bandit.opt_r
            else:
                if np.random.rand() < 0.0001:
                    bandit = en.Bandit(k)
                    # for agent in agent_list:
                    agent.opt_r = bandit.opt_r

        # for i, agent in enumerate(agent_list):
        # 腕の選択
        selected = agent.select_arm()
        # 報酬の観測
        reward = bandit.get_reward(int(selected))
        # 価値の更新
        agent.update(selected, reward)
        # accuracy
        accuracy[step] += bandit.get_correct(selected)
        # regret
        regret_sums += bandit.get_regret(selected)
        regrets[step] += regret_sums

    #print('finished')

    return regrets

def wrapper(args):
    # argsは(i, 2, 5)となっている
    return simulation(*args)


def multi_process(sampleList):
    # プロセス数:8(8個のcpuで並列処理)
    p = Pool(multiprocessing.cpu_count())
    #print(sampleList)
    output = p.map(wrapper, sampleList)
    # プロセスの終了
    p.close()
    return output

#simulation(10, 1000, 20, 1, 1, 1)

args = sys.argv
is_nonstationary = int(args[1])
is_constant = int(args[2])
print("is_nos:"+str(is_nonstationary))
print("is_cons:"+str(is_constant))

sim_num = 100
dir_csv_name = "csv/"
make_dir("/"+dir_csv_name)
agent_regrets = {}

#time calcu
start = time.time()

args_list = ["UCB1T", "TS", "RS_gamma", "RS_OPT", "TS_gamma", "RS_gamma2", "RS_OPT_gamma", "meta_UCB1T", "meta_TS",
             "meta_RS_gamma", "meta_RS_OPT"]
for x in range(len(args_list)):
    arg_list = [(args_list[x], 100000, 20, is_nonstationary, is_constant, 1, i) for i in range(sim_num)]
#alg, step_num, k, is_nonstationary, is_constant, switch, seed
    output = multi_process(arg_list)
    regrets = np.mean(output, 0)
    agent_regrets.update({args_list[x]:regrets})
    #print(output)


df = pd.DataFrame(agent_regrets)
if is_nonstationary == 1:
    str1 = 'nonstationary_'
else:
    str1 = 'stationary_'
if is_constant == 1:
    str2 = 'constant_'
else:
    str2 = 'nonconstant_'
print(dir_csv_name+str1+str2+'regrets.csv')
df.to_csv(dir_csv_name+str1+str2+'regrets.csv')


elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")