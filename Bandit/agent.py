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


class PS(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.k = k
        # sorted_index = np.argsort(bandit.probability)[::-1]
        # self.r = (bandit.probability[sorted_index[0]] + bandit.probability[sorted_index[1]]) / 2

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        if np.max(self.arm_values) > self.opt_r:
            return greedy(self.arm_values)
        else:
            return random.randint(self.k)

    def update(self, selected, reward):
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_values[selected] = self.arm_rewards[selected] / self.arm_counts[selected]


class RS_OPT(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.reward_sum = 0
        self.count = 0

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        return greedy(self.arm_values)

    def update(self, selected, reward):
        self.count += 1
        self.reward_sum += reward
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        average = self.arm_rewards[selected] / self.arm_counts[selected]
        self.arm_values[selected] = self.arm_counts[selected] * (average - self.opt_r)


class RS(Agent):
    def __init__(self, k, alpha=0.0005):
        super().__init__(k)
        self.r = 1
        self.alpha = alpha
        self.reward_sum = 0
        self.count = 0

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        return greedy(self.arm_values)

    def update(self, selected, reward):
        self.reward_sum += reward
        self.count += 1
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        average = self.arm_rewards[selected] / self.arm_counts[selected]
        self.arm_values[selected] = self.arm_counts[selected] * (average - self.r)
        self.r += self.alpha * (average - self.r)


class RS_gamma(Agent):
    def __init__(self, k, alpha=0.001, gamma=0.999):
        super().__init__(k)
        self.r = 1.0
        self.alpha = alpha
        self.reward_sum = 0
        self.count = 0
        self.gamma = gamma
        self.E = np.zeros(k)
        self.arm_w_counts = np.zeros(k) + 0.0001
        self.arm_l_counts = np.zeros(k) + 0.0001

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        # self.E = self.arm_w_counts / (self.arm_w_counts + self.arm_l_counts)
        self.arm_values = (self.arm_w_counts + self.arm_l_counts) * (self.E - self.r)
        return greedy(self.arm_values)

    def update(self, selected, reward):
        self.count += 1
        self.reward_sum += reward
        self.arm_w_counts *= self.gamma
        self.arm_l_counts *= self.gamma
        self.arm_w_counts[selected] += reward
        self.arm_l_counts[selected] += 1 - reward
        self.E[selected] = self.arm_w_counts[selected] / (self.arm_w_counts[selected] + self.arm_l_counts[selected])
        self.r += self.alpha * (self.E[selected] - self.r)


class RS_OPT_gamma(Agent):
    def __init__(self, k, gamma=0.999):
        super().__init__(k)
        self.gamma = gamma
        self.E = np.zeros(k)
        self.arm_w_counts = np.zeros(k) + 0.0001
        self.arm_l_counts = np.zeros(k) + 0.0001

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        self.E = self.arm_w_counts / (self.arm_w_counts + self.arm_l_counts)
        self.arm_values = (self.arm_w_counts + self.arm_l_counts) * (self.E - self.opt_r)
        return greedy(self.arm_values)

    def update(self, selected, reward):
        self.arm_w_counts *= self.gamma
        self.arm_l_counts *= self.gamma
        self.arm_w_counts[selected] += reward
        self.arm_l_counts[selected] += 1 - reward


class RS_CH(Agent):
    def __init__(self, k):
        self.arm_num = k
        self.arm_counts = np.ones(k) / 1000
        self.arm_count =np.sum(self.arm_counts)
        self.arm_rewards = np.zeros(k)
        self.arm_values = np.ones(k) / 2

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        g = greedy(self.arm_values)
        u_array = np.array([])
        r_array = np.array([])
        exceed_indices = []
        f = True

        # for j in range(self.arm_num):
        #     if j == g:
        #         u_array = np.append(u_array, -float('inf'))
        #         r_array = np.append(r_array, -float('inf'))
        #         continue
        #
        #     u = math.exp(-self.arm_counts[j] * ((1 - self.arm_values[j]) * math.log((1 - self.arm_values[j]) / (1 - self.arm_values[g]))) \
        #                                            + self.arm_values[j] * math.log((self.arm_values[j]) / (self.arm_values[g])))
        #
        #     if u == 1:
        #         u_array = np.append(u_array, -float('inf'))
        #         r_array = np.append(r_array, -float('inf'))
        #         continue
        #
        #     r = self.arm_values[g] * ((1 - (self.arm_values[j] / self.arm_values[g]) * u) / (1 - u))
        #
        #     rs_g = (self.arm_counts[g] / self.arm_count) * (self.arm_values[g] - r)
        #     rs_j = (self.arm_counts[j] / self.arm_count) * (self.arm_values[j] - r)
        #
        #     u_array = np.append(u_array, u)
        #     r_array = np.append(r_array, r)
        #     f = False
        #
        #     if rs_g > rs_j:
        #         exceed_indices.append(j)
        #
        # print(u_array)
        # if f:
        #     return np.random.randint(self.arm_num)
        #
        # if len(exceed_indices) == 0:
        #     return g
        # elif len(exceed_indices) == 1:
        #     return exceed_indices[0]
        # else:
        #     exceed_u_array = u_array[exceed_indices]
        #     max_u = np.amax(exceed_u_array)
        #     exceed_indices = np.array(exceed_indices)
        #     max_u_indices = exceed_indices[list(np.where(exceed_u_array == max_u)[0])]
        #     if len(max_u_indices) == 1:
        #         return max_u_indices[0]
        #     else:
        #         exceed_r_array = r_array[list(max_u_indices)]
        #         max_r = np.amax(exceed_r_array)
        #         max_r_indices = max_u_indices[list(np.where(exceed_r_array == max_r)[0])]
        #
        #         if len(max_r_indices) == 1:
        #             return max_r_indices[0]
        #         else:
        #             return max_r_indices[np.random.randint(len(max_r_indices))]

        # --------------------------------------------------------------------------------------------------------------

        for j in range(self.arm_num):
            if j == g:
                u_array = np.append(u_array, -float('inf'))
                r_array = np.append(r_array, -float('inf'))
                continue

            u = math.exp(-self.arm_counts[j] * ((1 - self.arm_values[j]) * math.log((1 - self.arm_values[j]) / (1 - self.arm_values[g]))) \
                                                   + self.arm_values[j] * math.log(self.arm_values[j] / self.arm_values[g]))

            if u == 1:
                u_array = np.append(u_array, -float('inf'))
                r_array = np.append(r_array, -float('inf'))
                continue

            u_array = np.append(u_array, u)

            r = self.arm_values[g] * ((1 - (self.arm_values[j] / self.arm_values[g]) * u) / (1 - u))
            r_array = np.append(r_array, r)
            f = False

        if f:
            return np.random.randint(self.arm_num)

        r_max = np.amax(r_array)

        for j in range(self.arm_num):
            if j == g:
                continue

            rs_g = (self.arm_counts[g] / self.arm_count) * (self.arm_values[g] - r_max)
            rs_j = (self.arm_counts[j] / self.arm_count) * (self.arm_values[j] - r_max)

            if rs_j > rs_g:
                exceed_indices.append(j)

        if len(exceed_indices) == 0:
            return g
        elif len(exceed_indices) == 1:
            return exceed_indices[0]
        else:
            exceed_u_array = u_array[exceed_indices]
            max_u = np.amax(exceed_u_array)
            exceed_indices = np.array(exceed_indices)
            max_u_indices = exceed_indices[list(np.where(exceed_u_array == max_u)[0])]
            if len(max_u_indices) == 1:
                return max_u_indices[0]
            else:
                exceed_r_array = r_array[list(max_u_indices)]
                max_r = np.amax(exceed_r_array)
                max_r_indices = max_u_indices[list(np.where(exceed_r_array == max_r)[0])]

                if len(max_r_indices) == 1:
                    return max_r_indices[0]
                else:
                    return max_r_indices[np.random.randint(len(max_r_indices))]

    def update(self, selected, reward):
        self.arm_count += 1
        a = 1 / (1 + self.arm_counts[selected])
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_values[selected] = (1 - a) * self.arm_values[selected] + a * reward


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


class TS_gamma(Agent):
    def __init__(self, k, gamma=0.999):
        super().__init__(k)
        self.sampling = np.ones((k, 2))
        self.gamma = gamma
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
        self.arm_w_counts *= self.gamma
        self.arm_l_counts *= self.gamma
        self.arm_w_counts[selected] += reward
        self.arm_l_counts[selected] += 1 - reward
        self.arm_rewards[selected] += reward
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


class meta_bandit(Agent):
    def __init__(self, k, agent, higher_agent, l=500, delta=0, lmd=30):
        super().__init__(k)
        self.original_agent = agent
        self.original_higher_agent = higher_agent
        self.old_agent = cp.deepcopy(self.original_agent)
        self.new_agent = cp.deepcopy(self.original_agent)
        self.higher_agent = cp.deepcopy(self.original_higher_agent)
        self.meta_flag = 0
        self.l = l
        self.l_count = 0
        self.delta = delta
        self.lmd = lmd
        self.step = 0
        self.selected = 0
        self.k = k
        self.mT_sum = 0
        self.MT = 0

    def reset_params(self):
        self.meta_flag = 0
        self.l_count = 0
        self.mT_sum = 0
        self.MT = 0

    def select_arm(self):
        self.old_agent.opt_r = self.opt_r
        self.new_agent.opt_r = self.opt_r
        self.higher_agent.opt_r = self.opt_r

        if self.meta_flag:
            self.selected = self.higher_agent.select_arm()

            if self.selected == 0:
                return self.old_agent.select_arm()
            else:
                return self.new_agent.select_arm()
        else:
            return self.old_agent.select_arm()

    def update(self, selected, reward):
        self.step += 1

        if self.meta_flag:
            self.l_count += 1
            self.higher_agent.update(self.selected, reward)
            self.old_agent.update(selected, reward)
            self.new_agent.update(selected, reward)

            if self.l_count == self.l:
                self.reset_params()

                if greedy(self.higher_agent.arm_rewards) == 1:
                    self.old_agent = self.new_agent

        else:
            self.old_agent.update(selected, reward)
            # max_arm = greedy(self.old_agent.arm_rewards)
            # rt_mean =  self.old_agent.arm_rewards[max_arm] /  self.old_agent.arm_counts[max_arm]
            rt_mean = self.old_agent.reward_sum / self.old_agent.count
            mT = reward - rt_mean + self.delta
            self.mT_sum += mT
            if self.mT_sum > self.MT:
                self.MT = self.mT_sum
            PHT = self.MT - self.mT_sum

            if PHT > self.lmd:
            # if self.step % 10000 == 0:
            #     print(self.step)
                self.meta_flag = 1
                self.new_agent = cp.deepcopy(self.original_agent)
                self.higher_agent = cp.deepcopy(self.original_higher_agent)
