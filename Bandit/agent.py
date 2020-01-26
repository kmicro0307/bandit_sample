import numpy as np
import copy as cp
import math
import random


def greedy(values):
    max_values = np.where(values == values.max())
    return np.random.choice(max_values[0])


class Agent():
    def __init__(self, k):
        self.arm_counts = np.zeros(k)
        self.arm_rewards = np.zeros(k)
        self.arm_values = np.zeros(k)
        self.opt_r = 0

    def recet_params(self):
        self.value = np.zeros_like(self.value)

    def select_arm(self):
        pass

    def update(self, selected, reward):
        pass


class TS(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.sampling = np.ones((k, 2))
        self.alpha = 1
        self.beta = 1
        self.reward_sum = 0
        self.count = 0
        self.arm_w_counts = np.zeros(k)
        self.arm_l_counts = np.zeros(k)

    def recet_params(self):
        super().recet_params()

    def select_arm(self):
        return np.argmax([np.random.beta(n[0], n[1], 1) for n in self.sampling])

    def update(self, selected, reward):
        self.count += 1
        self.reward_sum += reward
        self.arm_w_counts[selected] += reward
        self.arm_l_counts[selected] += 1 - reward
        self.sampling[selected] = (self.arm_w_counts[selected]+self.alpha, self.arm_l_counts[selected]+self.beta)


class UCB1T(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.f = 0
        self.reward_sum = 0
        self.arm_rewards_square = np.zeros(k)
        self.k = k
        self.count = 0

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        if self.f == 0 and np.amin(self.arm_counts) == 0:
            return np.argmin(self.arm_counts)
        else:
            self.f = 1
            return greedy(self.arm_values)

    def update(self, selected, reward):
        self.count += 1
        self.reward_sum += reward
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_rewards_square[selected] += reward * reward

        # if self.count >= self.k:
        if self.f == 1:
            for i in range(0, self.k):
                ave = self.arm_rewards[i] / self.arm_counts[i]
                variance = self.arm_rewards_square[i] / self.arm_counts[i] - ave * ave
                v = variance + math.sqrt((2.0 * math.log(self.count)) / self.arm_counts[i])
                self.arm_values[i] = ave + math.sqrt((math.log(self.count) / self.arm_counts[i]) * min(0.25, v))
