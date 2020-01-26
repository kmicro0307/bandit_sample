import csv
import time
import os
import sys
import multiprocessing
from collections import Counter
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import environment as en
import agent as ag


class Simlations():

    def __init__(self, env):
        self.sims = {"sim_normal": self.simulation,
                     "sim_switch": self.simulation_swich,
                    }
        self.sim = self.sims[env]

    def simulation(self, args):
        alg, step_num, k, is_nonstationary, is_constant, switch, seed = args
        np.random.seed(seed=seed)

        accuracy = np.zeros(step_num)
        regrets = np.zeros(step_num)

        bandit = en.Bandit(k)
        agent_list = {"UCB1T": ag.UCB1T(k),
                      "TS": ag.TS(k),
                      }

        agent = agent_list[alg]
        agent.opt_r = bandit.opt_r
        regret_sums = 0

        for step in range(step_num):

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

        return regrets

    def simulation_swich(self, args):
        alg, step_num, k, is_nonstationary, is_constant, switch, seed = args
        np.random.seed(seed=seed)

        p_switch = 1/switch

        accuracy = np.zeros(step_num)
        regrets = np.zeros(step_num)

        bandit = en.Bandit(k)
        agent_list = {"UCB1T": ag.UCB1T(k),
                      "TS": ag.TS(k),
                      }

        agent = agent_list[alg]
        agent.opt_r = bandit.opt_r
        regret_sums = 0

        for step in range(step_num):

            # バンディット変化
            if is_nonstationary:
                if is_constant:
                    if step % switch == 0:
                        bandit = en.Bandit(k)
                        # for agent in agent_list:
                        agent.opt_r = bandit.opt_r
                else:
                    if np.random.rand() < p_switch:
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

        return regrets

    def simulation_swichtest(self, args):
        # optimal wo erabanakatta kakuritu
        alg, step_num, k, is_nonstationary, is_constant, switch, seed = args
        np.random.seed(seed=seed)

        p_switch = 1/switch

        accuracy = np.zeros(step_num)
        regrets = np.zeros(step_num)

        bandit = en.Bandit(k)
        agent_list = {"UCB1T": ag.UCB1T(k),
                      "TS": ag.TS(k),
                      }

        agent = agent_list[alg]
        agent.opt_r = bandit.opt_r
        regret_sums = 0

        for step in range(step_num):

            # バンディット変化
            if is_nonstationary:
                if is_constant:
                    if step % switch == 0:
                        bandit = en.Bandit(k)
                        # for agent in agent_list:
                        agent.opt_r = bandit.opt_r
                        #agent.get_switch(k)
                else:
                    if np.random.rand() < p_switch:
                        bandit = en.Bandit(k)
                        # for agent in agent_list:
                        agent.opt_r = bandit.opt_r

            # for i, agent in enumerate(agent_list):
            # 腕の選択
            selected = agent.select_arm()
            # 報酬の観測
            agent.count_arm(step, selected=selected)
            reward = bandit.get_reward(int(selected))
            # 価値の更新
            agent.update(selected, reward)
            # accuracy
            accuracy[step] += bandit.get_correct(selected)
            # regret
            regret_sums += bandit.get_regret(selected)
            regrets[step] += regret_sums

        return agent.arm_selected
