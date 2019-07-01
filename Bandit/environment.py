import numpy as np


class Bandit(object):
    def __init__(self, k):
        while True:
            self.probability = np.asarray(np.random.rand(k))
            if len(self.probability) == len(set(self.probability)):
                break

        self.correct_arm = np.argmax(self.probability)
        self.max_prob_arm = np.amax(self.probability)
        sorted_prob = np.sort(self.probability)[::-1]
        self.opt_r = (sorted_prob[0] + sorted_prob[1]) / 2

    #         print(self.probability)

    def get_reward(self, selected):
        if self.probability[selected] >= np.random.rand():
            return 1
        else:
            return 0

    def get_correct(self, selected):
        if self.correct_arm == selected:
            return 1
        else:
            return 0

    def get_regret(self, selected):
        return self.max_prob_arm - self.probability[selected]