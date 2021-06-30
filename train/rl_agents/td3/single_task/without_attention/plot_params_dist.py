"""
Management methods for traffic flow params generation.
"""

import os

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
import pickle
from interval import Interval


def get_dist(file_path):
    """

    :return:
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # result_dict = data_dict,
    # save_result = True,
    # plot_fig = True,
    # save_path = save_path,

    # for each traffic flow
    for key, item in data_dict.items():
        speed_list = []
        distance_list = []
        #
        total_vehicle_num = len(item)
        x = list(range(total_vehicle_num))
        #
        for tup in item:
            speed_list.append(tup[0])
            distance_list.append(tup[1])

        speed_range = Interval(10, 20)

        filtered_speed_list = []
        filtered_dist_list = []

        for index, speed in enumerate(speed_list):

            if speed not in speed_range:
                continue

            filtered_speed_list.append(speed)
            filtered_dist_list.append(distance_list[index])

        # plot the scatter
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('speed')
        ax.set_ylabel('distance')

        ax.scatter(filtered_speed_list, filtered_dist_list, c='b', s=15, alpha=0.5)
        # ax.scatter(collision_velo, collision_dist, c='r', s=15, alpha=0.5)
        # ax.scatter(time_exceed_velo, time_exceed_dist, c='b', s=15, alpha=0.5)

        plt.show()

        plt.figure()
        plt.hist(np.array(filtered_dist_list), density=True, bins=50)
        plt.show()


if __name__ == '__main__':

    # stat results
    path = '/outputs/right/rl_results'
    file_path = os.path.join(path, 'tf_distribution.pkl')
    get_dist(file_path)
