import os
import seaborn as sns
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

from gym_carla.modules.trafficflow.ou_noise import stat_tf_distribution


# path = 'mean_rewards_data.npy'
# reward_data = np.load(path)
# y_data = reward_data.tolist()
# x_data = np.arange(1, len(y_data)+1, 1)

# plt.plot(x_data, y_data)
# plt.show()

# mean_path = 'mean_rewards_data.npy'
# mean_data = np.load(mean_path)
# std_path = 'mean_rewards_data.npy'
# std_data = np.load(std_path)
# d = {"mean": mean_data}
# with open(os.path.join("dqn_reward_data_"+".pkl"), "wb") as f:
# pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

result_folder = ''

# file = "reward_data"+".pkl"
reward_file = os.path.join(result_folder, 'reward_data.pkl')
with open(os.path.join(reward_file), "rb") as f:
    data = pickle.load(f)
x1 = data["avarage_rewards"]
x1 = x1.T

time = []
for i in range(x1.shape[0]):
    time.append(i)

fig1 = plt.figure()
sns.lineplot(x=time, y=x1)
plt.ylabel("Rewards")
plt.xlabel("Episode")
plt.title("td3")

# save figure
plt.savefig(os.path.join(result_folder, "Rewards.png"))
plt.show()

# file = "success_rate_data"+".pkl"
success_rate_file = os.path.join(result_folder, 'success_rate_data.pkl')
with open(os.path.join(success_rate_file), "rb") as f:
    data = pickle.load(f)
x2 = data["recent_success_rates"]
x2 = x2.T

display_types = ['reward', 'success rate']
display_type = display_types[1]

sns.set(style="darkgrid", font_scale=1.5)


fig2 = plt.figure()
sns.lineplot(x=time, y=x2)
plt.ylabel("Success rates")
plt.xlabel("Episode")
plt.title("td3")

# save figure
plt.savefig(os.path.join(result_folder, "success_rate.png"))
plt.show()


# if display_type == 'reward':
#     sns.tsplot(time=time, data=x1, color="r", condition="Reward curve")
#     plt.ylabel("Rewards")
# else:
#     sns.tsplot(time=time, data=x2, color="b", condition="Success rate curve")
#     plt.ylabel("Success rates")


# plt.xlabel("Episode")
# plt.title("ddpg with safety layer")

# with attention
# plt.title("ddpg with attention and safety layer")

# plt.show()

print('')

# ================================
# plot distribution of the traffic flow params

# tf_file = os.path.join(result_folder, 'tf_distribution.pkl')
# with open(os.path.join(tf_file), "rb") as f:
#     data = pickle.load(f)
#
# save_path = os.path.join(result_folder, 'tf_dist')
#
# stat_tf_distribution(result_dict=data,
#                      save_result=True,
#                      plot_fig=True,
#                      save_path=save_path,
#                      )


