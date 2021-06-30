"""
Config for carla simulation.
"""
import os
import sys

import numpy as np


version_config = {

    # ================   CARLA simulator   ================

    # 'carla_version': '0.9.9',
    'carla_version': '0.9.10.1',
    # 'carla_version': '0.9.10.1_rss',  # todo test rss usage
    # 'carla_version': '0.9.11',


    # todo fix relative path, find CARLA_simulator folder
    # 'root_path': '/home/lyq/CARLA_simulator',

    # on dell PC
    'root_path': '/home/lyq/CARLA_simulator',

    # on 1660ti server
    # 'root_path': '/home/liuyuqi/CARLA_simulator',

    # on the 2080ti server
    # 'root_path': '/home1/lyq/CARLA_simulator',
    # user sim
    # 'root_path': '/home1/sim/CARLA_simulator',

    # ================   project parent folder   ================

    # Dell PC
    # 'parent_folder': '/home/lyq/PycharmProjects/',

    # 1660ti
    # 'parent_folder': '/home/liuyuqi/PycharmProjects',

    # 2080ti server
    # 'parent_folder': '/home1/lyq/PycharmProjects/',
    # user sim
    # 'parent_folder': '/home1/sim/PycharmProjects/',

}


# todo in developing

# # start transform of ego vehicle
# start_transform = np.array([])

#
# co_simulation_params = {
#
#     # carla settings
#     '': ,
#     '': ,
#
#
#     #
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
#     '': ,
# }






