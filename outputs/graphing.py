import numpy as np
import matplotlib.pyplot as plt

regret = np.array([[0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 21],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 20],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 21],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 19, 20],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 20],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 20],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 19, 19],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 20],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 20],
                   [0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 8, 12, 16, 16, 19, 19, 19, 20, 21]])

combined_cumulative_regret = np.sum(regret, axis=0)
print(combined_cumulative_regret)

t = np.linspace(0, len(regret[0]), len(regret[0]), endpoint=False)

fig = plt.figure()
# for agent_regret in regret:
#     plt.plot(t, agent_regret)
plt.plot(t, combined_cumulative_regret)
plt.title("Multi-Agent UCB (Degree = 4)")
plt.xlabel("Timestep")
plt.ylabel("Cumulative Regret")
plt.xticks(np.arange(t[0], t[-1] + 1, 2.0))
# plt.yticks(np.arange(0, 25, 2.0))
plt.grid(True)
plt.show()

print(t)
