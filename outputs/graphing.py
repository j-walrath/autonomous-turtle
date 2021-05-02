import os
import numpy as np
import matplotlib.pyplot as plt

path = '.'
titles = []
regrets = []
for filename in os.listdir(path):
    if filename == 'graphing.py':
        continue
    with open(os.path.join(path, filename)) as f:
        titles.append(f.readline())
        regrets.append(np.loadtxt(f, delimiter=', ', skiprows=2, max_rows=10))

# rows = 4
# columns = 5
# fig, axs = plt.subplots(rows, columns, figsize=(20, 25))
# for row in range(rows):
#     for col in range(columns):
#         i = row * columns + col
#         regret = regrets[i]
#         combined_cumulative_regret = np.sum(regret, axis=0)
#         # print(combined_cumulative_regret)
#
#         t = np.linspace(0, len(regret[0]), len(regret[0]), endpoint=False)
#
#         # for agent_regret in regret:
#         #     plt.plot(t, agent_regret)
#         axs[row, col].plot(t, combined_cumulative_regret)
#         axs[row, col].set_title(titles[i][16:-1], fontsize=8)
#         # axs[row, col].set_xlabel("Timestep", fontsize=6)
#         # axs[row, col].set_ylabel("Cumulative Regret", fontsize=6)
#         # plt.xticks(np.arange(t[0], t[-1] + 1, 2.0))
#         # plt.yticks(np.arange(0, 25, 2.0))
#         # plt.grid(True)
#
# fig.suptitle('Multi-Agent UCB1')
# plt.setp(axs[-1, :], xlabel='Timestep')
# plt.setp(axs[:, 0], ylabel='Cumulative Regret')
# plt.show()

for i, regret in enumerate(regrets):
    if i in (0, 1, 8, 9, 18, 19):
        fig = plt.figure(figsize=(8, 5))
        t = np.linspace(0, len(regret[0]), len(regret[0]), endpoint=False)
        for agent_regret in regret:
            plt.plot(t, agent_regret)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Regret")
        plt.title(titles[i][16:-1])
        plt.legend(['Agent {}'.format(x) for x in range(10)], loc=2)

plt.show()

# total_regret = ([], [])
# timesteps = ([], [])
# for i, regret in enumerate(regrets):
#     idx = 0 if i & 1 else 1
#
#     total_regret[idx].append((np.sum(regret, axis=0))[-1])
#     timesteps[idx].append(np.size(regret, axis=1))

# print(total_regret)
# print(timesteps)

# fig = plt.figure(figsize=(8, 5))
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# plt.scatter(x, timesteps[0], marker='D')
# plt.scatter(x, timesteps[1], c='r', marker='D')
# plt.ylabel("Timesteps Taken to Clear Field")
# plt.xlabel("Agent Degree")
# plt.title("Collection Time vs. Agent Degree")
# plt.legend(["w/o message", "w/ message"])
# plt.show()
