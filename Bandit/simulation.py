import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag


def simulation(simulation_num, step_num, k, agent_num, is_nonstationary, is_constant, file_name, switch):
    labels = ["UCB1T", "TS", "RS", "RS OPT", "TS gamma", "RS gamma", "RS OPT gamma", "meta UCB1T", "meta TS", "meta RS", "meta RS OPT"]
    colors = ["g", "b", "r", "m", "c", "#ff7f00", "#a65628", "g", "b", "r", "m"]
    accuracy = np.zeros((agent_num, step_num))
    regrets = np.zeros((agent_num, step_num))

    for sim in range(simulation_num):
        print(sim + 1)
        bandit = en.Bandit(k)
        agent_list = [ag.UCB1T(k),
                      ag.TS(k),
                      ag.RS_gamma(k, gamma=1.0),
                      ag.RS_OPT(k),
                      ag.TS_gamma(k),
                      ag.RS_gamma(k),
                      ag.RS_OPT_gamma(k),
                      ag.meta_bandit(k, agent=ag.UCB1T(k), higher_agent=ag.UCB1T(2), l=500, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.TS(k), higher_agent=ag.TS(2), l=30, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.RS_gamma(k, gamma=1.0), higher_agent=ag.RS_gamma(2, gamma=1.0), l=30, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.RS_OPT(k), higher_agent=ag.RS_OPT(2), l=30, delta=0, lmd=30)]
        regret_sums = np.zeros(agent_num)

        for agent in agent_list:
            agent.opt_r = bandit.opt_r
        regret_sums = np.zeros(agent_num)
        # prev_selecteds = np.zeros(agent_num)

        for step in range(step_num):
            prev_selected = 0
            regret = 0

            # バンディット変化
            if is_nonstationary:
                if is_constant:
                    if step % 10000 == 0:
                        bandit = en.Bandit(k)
                        for agent in agent_list:
                            agent.opt_r = bandit.opt_r
                else:
                    if np.random.rand() < 0.0001:
                        bandit = en.Bandit(k)
                        for agent in agent_list:
                            agent.opt_r = bandit.opt_r

            for i, agent in enumerate(agent_list):
                # 腕の選択
                selected = agent.select_arm()
                # 報酬の観測
                reward = bandit.get_reward(int(selected))
                # 価値の更新
                agent.update(selected, reward)
                # accuracy
                accuracy[i][step] += bandit.get_correct(selected)
                # regret
                regret_sums[i] += bandit.get_regret(selected)
                regrets[i][step] += regret_sums[i]

    accuracy /= simulation_num
    regrets /= simulation_num

    plt.xlabel('steps')
    plt.ylabel('accuracy')
    # plt.xscale("log")
    plt.ylim([0.0, 1.1])
    for i, graph in enumerate(accuracy):
        plt.plot(graph, label=labels[i], color=colors[i])
    plt.legend(loc="best")
    plt.savefig("stationary_accuracy")
    # plt.show()
    plt.cla()

    plt.xlabel('steps')
    plt.ylabel('regret')
    # plt.xscale("log")
    for i, graph in enumerate(regrets):
        plt.plot(graph, label=labels[i], color=colors[i])
    plt.legend(loc="best")
    plt.savefig("stationary_regret")
    # plt.show()
    plt.cla()

    file_data = np.concatenate([accuracy, regrets], 0)

    accuracy_header = "UCB1T accuracy,TS accuracy,RS accuracy,RS OPT accuracy,TS gamma accuracy,RS gamma accuracy,RS OPT gamma accuracy,meta UCB1T accuracy,meta TS accuracy,meta RS accuracy,meta RS OPT accuracy"
    regret_header = ",UCB1T regret,TS regret,RS regret,RS OPT regret,TS gamma regret,RS gamma regret,RS OPT gamma regret,meta UCB1T regret,meta TS regret,meta RS regret,meta RS OPT regret"
    np.savetxt(file_name, file_data.transpose(), delimiter=",", header=accuracy_header+regret_header)



simulation(100, 100000, 20, 11, 1, 1, "stationary.csv", 1)
