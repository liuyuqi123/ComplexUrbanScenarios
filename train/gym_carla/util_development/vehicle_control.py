"""
Some planners and controller APIs in this script.

Rule-based Planners and controllers for a carla vehicle.
"""

import carla

from train.gym_carla.navigation.global_route_planner import GlobalRoutePlanner
from train.gym_carla.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from train.gym_carla.navigation.controller import VehiclePIDController

import numpy as np


class PathPlanner(object):
    """ Generate a route (a series of waypoints) to the goal, please use this in 'reset' """

    def __init__(self, carla_map):
        # sampling_resolution is the sampling distance between waypoints
        dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution=1.0)
        self.grp = GlobalRoutePlanner(dao)
        self.grp.setup()

    def generate_waypoints(self, ego_car, goal_loc):
        """
        Return a route(waypoint list) to the goal, using A* algorithm

        ego_car: carla.vehicle
        goal_loc: carla.Location or ndarray
        """
        # check goal_loc
        if isinstance(goal_loc, carla.Location):
            goal_loc.z = 0.5
        elif isinstance(goal_loc, np.ndarray):
            goal_loc = carla.Location(x=goal_loc[0], y=goal_loc[1], z=0.5)
        else:
            assert 'wrong datatype of goal_loc.'

        begin = ego_car.get_location()
        end = goal_loc
        route_trace = self.grp.trace_route(begin, end)

        waypoint_list = []
        for item in route_trace:  # each item is (carla.Waypoint, RoadOption)
            waypoint_list.append((item[0]))

        return waypoint_list

    def buffer_waypoints(self):
        """buffer near waypoint for ego vehicle."""

        # todo package waypoints buffer into the manager
        # self.waypoint_buffer
        pass


class Controller:
    """
    An extra class to contain PID controller, for adjusting parameters.

    Generate basic control command to follow a series of waypoints, please use this in 'step'.
    """

    def __init__(self, vehicle, dt):
        """
        Init with all parameters.
        :param vehicle: vehicle to be controlled.
        :param dt: maximum timestep gap
        """
        self.vehicle = vehicle
        args_lateral_dict = {'K_P': 1.95,
                             'K_D': 0.2,
                             'K_I': 0.07,
                             'dt': dt}

        args_longitudinal_dict = {'K_P': 1.0,
                                  'K_D': 0,
                                  'K_I': 0.05,
                                  'dt': dt}

        # register the vehicle
        self.pid_controller = VehiclePIDController(self.vehicle,
                                                   args_lateral=args_lateral_dict,
                                                   args_longitudinal=args_longitudinal_dict,
                                                   max_throttle=1.0,  # 0.75,
                                                   max_brake=1.0,  # 0.75
                                                   max_steering=0.5  # 0.75
                                                   )

    def generate_control(self, target_speed, next_waypoint):
        """
        Return a basic control command

        :param target_speed: speed in m/s
        :param next_waypoint: using carla waypoint class
        :return:
        """
        car_control = self.pid_controller.run_step(target_speed, next_waypoint)
        return car_control
