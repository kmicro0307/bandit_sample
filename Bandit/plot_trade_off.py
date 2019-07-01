import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15), dpi=50)

#ファイルの読み込み。最後だけ
st = pd.read_csv('stationary.csv')[99998:-1]
non_st = pd.read_csv('nonstationary.csv')[99998:-1]
# non_st_const = pd.read_csv('nonsteady_constant.csv')[99998:-1]
# st = pd.read_csv('test.csv')[99998:-1]
# non_st = pd.read_csv('test1.csv')[99998:-1]

#いらない列の削除
# drop_column = ["UCB1T regret", "TS regret", "RS regret", "RS OPT regret", "TS gamma regret", "RS gamma regret", "RS OPT gamma regret", "meta UCB1T regret", "meta TS regret", "meta RS regret", "meta RS OPT regret"]
drop_column = ["# UCB1T accuracy", "TS accuracy", "RS accuracy", "RS OPT accuracy", "TS gamma accuracy", "RS gamma accuracy", "RS OPT gamma accuracy", "meta UCB1T accuracy", "meta TS accuracy", "meta RS accuracy", "meta RS OPT accuracy"]
# drop_column = ["UCB1T accuracy", "RS accuracy", "RS OPT accuracy", "RS gamma accuracy", "RS OPT gamma accuracy", "meta UCB1T accuracy", "meta RS accuracy", "meta RS OPT accuracy"]
st = st.drop(drop_column, axis=1)
non_st = non_st.drop(drop_column, axis=1)
# non_st_const = non_st_const.drop(drop_column, axis=1)

print(st)

#平均の計算
# st_means = st.mean()[99900:-1]
# non_st_means = non_st.mean()[99900:-1]
# non_st_const_means = non_st_const.mean()[99900:-1]
st_means = st
non_st_means = non_st
# non_st_const_means = non_st_const

names = st.columns.values
print(names)

#グラフの描画

fig = plt.figure(figsize=(8,6), dpi=100)

ax = fig.add_subplot(1,1,1)

labels = ["UCB1T", "TS", "RS", "RS OPT", "TS gamma", "RS gamma", "RS OPT gamma", "meta UCB1T", "meta TS", "meta RS", "meta RS OPT"]
colors = ["g", "b", "r", "m", "c", "#ff7f00", "#a65628", "g", "b", "r", "m"]
markers = ["o", "s", "^", "v", "<", ">", "p", "D", "x", "1", "2"]
# colors = ["g", "g", "g", "g", "b", "b", "b", "r", "r", "r", "r"]
# markers = ["o", "s", "^", "v", "s", "^", "v", "o", "s", "^", "v"]
# 非同期の点をうつ
# for i,(x,y) in enumerate(zip(st_means , non_st_means)):
#     #ax.annotate(names[i],(x,y))
#     ax.annotate("no sync",(x+0.001,y-0.04))
#     plt.plot(x, y, marker=markers[i], ms=10, color=colors[i])


# 同期の点をうって線で繋ぐ
for i in range(len(names)):
    x = st_means[names[i]]
    y = non_st_means[names[i]]
    dx = 0
    dy = non_st_means[names[i]] - y
    # plt.xscale("log")
    # plt.yscale("log")
    # ax.arrow(x, y, dx, dy, width=0.0001, color='black', head_length=0.0, head_width=0.0)
    # plt.xlim([0.4, 1.0])
    # plt.yticks([1E1, 1E2, 1E3, 1E4])
    plt.plot(x, y, label=labels[i], marker=markers[i], ms=10, color=colors[i])
    plt.legend(loc="upper right", frameon=False)
    # ax.annotate("sync", (x+0.001, y+0.015))
    # ax.annotate(labels[i], (x+0.001, y+0.015))

# for i,(x,y) in enumerate(zip(st_means , non_st_means)):
#     ax.annotate("no sync",(x,y-0.03))

plt.xlabel("stationary")
plt.ylabel("non_stationary")
plt.savefig("trade_off")
# plt.show()