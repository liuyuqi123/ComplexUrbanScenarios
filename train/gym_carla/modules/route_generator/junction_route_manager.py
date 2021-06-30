"""
This method is not fully finished yet.

This is the route generator for the traffic flow
planned and controlled by user manually.

"""

from train.gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

# ==================================================
# import carla module
import glob
import os
import sys

carla_root = os.path.join(root_path, 'CARLA_' + carla_version)
carla_path = os.path.join(carla_root, 'PythonAPI')
sys.path.append(carla_path)
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla'))
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/agents'))

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from train.gym_carla.envs.BasicEnv import BasicEnv
from train.gym_carla.modules.carla_module import CarlaModule

from train.gym_carla.util_development.scenario_helper_modified import (generate_target_waypoint,
                                                                 get_waypoint_in_distance
                                                                 )
from train.gym_carla.util_development.route_manipulation import interpolate_trajectory

from train.gym_carla.util_development.util_visualization import draw_waypoint
from train.gym_carla.util_development.carla_color import *

import numpy as np
import random

# Town03 default junction center
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)


class TrafficFlowRouteManager(CarlaModule):

    # dict to store all information
    route_info = {
        'positive_x': {
            'spawn_point': None,
            'reach_distance': None,
            'sink_point': None,
            'departure_distance': None,
        },

        'negative_x': {
            'spawn_point': None,
            'reach_distance': None,
            'sink_point': None,
            'departure_distance': None,
        },

        'negative_y_0': {
            'spawn_point': None,
            'reach_distance': None,
            'sink_point': None,
            'departure_distance': None,
        },

        'negative_y_1': {
            'spawn_point': None,
            'reach_distance': None,
            'sink_point': None,
            'departure_distance': None,
        },

        'positive_y_0': {
            'spawn_point': None,
            'reach_distance': None,
            'sink_point': None,
            'departure_distance': None,
        },

        'positive_y_1': {
            'spawn_point': None,
            'reach_distance': None,
            'sink_point': None,
            'departure_distance': None,
        },

    }

    def __init__(self,
                 carla_api,
                 junction=None,
                 ):
        
        super(TrafficFlowRouteManager, self).__init__(carla_api)

        if junction:
            self.junction = junction
        else:
            self.junction = self.get_junction_by_location(junction_center)

        self.entrance_waypoints = []
        self.exit_waypoints = []

    def set_junction(self, junction: carla.Junction):
        """
        Setter method to set a junction.
        """
        self.junction = junction

    def get_junction_waypoints(self):
        """
        Get all available entrance and exit waypoints of a junction.
        """
        if not self.junction:
            raise ValueError("Junction is not assigned or wrong type, please check.")

        f = isinstance(self.junction, carla.Junction)

        lane_type = carla.LaneType.Driving
        wp_list = self.junction.get_waypoints(lane_type)

        # filter entrance and exit waypoints
        entrance_waypoints = []
        exit_waypoints = []

        for entrance_wp, exit_wp in wp_list:
            # check and append in entrance
            in_entrance_flag = False
            for wp in entrance_waypoints:
                dist = entrance_wp.transform.location.distance(wp.transform.location)
                if dist <= 0.5:
                    in_entrance_flag = True

            if not in_entrance_flag:
                entrance_waypoints.append(entrance_wp)

            # check and append in exit
            in_exit_flag = False
            for wp in exit_waypoints:
                dist = exit_wp.transform.location.distance(wp.transform.location)
                if dist <= 0.5:
                    in_exit_flag = True

            if not in_exit_flag:
                exit_waypoints.append(exit_wp)

        self.entrance_waypoints = entrance_waypoints
        self.exit_waypoints = exit_waypoints

        return entrance_waypoints, exit_waypoints

    def get_route_waypoints(self):
        """
        Get start waypoints and end waypoints
        for traffic flow route.
        """
        hop_resolution = 1.0
        for entrance_wp in self.entrance_waypoints:
            draw_waypoint(self.world, entrance_wp, (magenta, magenta))

            _wp = entrance_wp.previous(1.)[0]
            # _wp_list = [_wp]
            # for i in range(10):
            #     _wp = _wp.previous(1.)[0]
            #     _wp_list.append(_wp)
            #     draw_waypoint(self.world, _wp, (red, red))

            _wp_list = _wp.previous_until_lane_start(hop_resolution)

            for wp in _wp_list:
                draw_waypoint(self.world, wp, (green, green))

            wp_list = entrance_wp.previous_until_lane_start(hop_resolution)
            wp_list_2 = entrance_wp.next_until_lane_end(hop_resolution)

            for wp in wp_list:
                draw_waypoint(self.world, wp, (red, red))

            for wp in wp_list_2:
                draw_waypoint(self.world, wp, (yan, yan))



            print('')

        print('')

    def get_sink_location(self):
        """
        A developing method.

        Get sink location of a route.
        :return:
        """
        #
        tf_directions = ['negative_x', 'negative_y_0']
        turn_flags = [0, -1]
        spawn_transforms = [
            carla.Transform(carla.Location(x=-64.222389, y=135.423065, z=0.000000),
                            carla.Rotation(pitch=0.000000, yaw=-361.296783, roll=0.000000)),


        ]

        for index in range(2):
            spawn_transform = spawn_transforms[index]
            turn_flag = turn_flags[index]



class TestRoute:

    def __init__(self):
        self.env = BasicEnv(
            port=2000,
            tm_port=8100,
            sync_mode=False,
        )

        self.carla_api = self.env.get_env_api()

        self.world = self.carla_api['world']


        self.env.set_spectator_overhead(junction_center, yaw=270, h=50)

    def test_1(self):
        trans = carla.Transform(carla.Location(x=5.854240, y=189.210159, z=0.000000),
                                carla.Rotation(pitch=0.000000, yaw=-90.362534, roll=0.000000))

        wp = self.env.map.get_waypoint(trans.location, project_to_road=True)

        draw_waypoint(self.world, trans, (blue, blue))

        print('')

    def run(self):

        route_manager = TrafficFlowRouteManager(self.carla_api)

        entrance_waypoints, exit_waypoints = route_manager.get_junction_waypoints()

        route_manager.get_route_waypoints()

        print('done.')


if __name__ == '__main__':

    test = TestRoute()
    # test.test_1()

    test.run()
