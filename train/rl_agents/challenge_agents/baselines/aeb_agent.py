"""
An agent controlled by local_planner combined with an AEB method.

This class is developed inherited from test_agent.

"""

import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque

import carla

from train.rl_agents.challenge_agents.test_agents.test_agent import TestAgent
from train.gym_carla.modules.trafficflow.traffic_flow_manager4 import detect_lane_obstacle


class AebAgent(TestAgent):

    def __init__(self, path_to_conf_file, target_speed=20):
        """
        A target speed for vehicle is supposed to be assigned.

        :param path_to_conf_file:
        :param target_speed: default target speed in km/h
        """
        
        super(AebAgent, self).__init__(path_to_conf_file)

        # in km/h
        self.target_speed = target_speed

    def load_model(self, route_option, manual=False, debug=False):
        """
        No need to load deep models.
        """
        pass

    def set_target_speed(self, target_speed):
        """
        Setter for target speed.
        """
        self.target_speed = target_speed

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.

        :return: carla.VehicleControl
        """
        # use state manager to get current state
        state = self.state_manager.get_state()
        self.state = state

        # get ego vehicle from state manager
        self.ego_vehicle = self.state_manager.get_ego_vehicle()

        # ==================================================
        # --------------  Get vehicle control  -------------
        # ==================================================
        self.buffer_waypoint()

        if self._waypoint_buffer:

            if detect_lane_obstacle(
                    world=self.world,
                    actor=self.ego_vehicle,
                    detect_ego_vehicle=False,
                    extension_factor=3.,
                    margin=1.02
            ):
                target_speed = 0.0
            else:
                target_speed = self.target_speed

            print('target speed: ', target_speed)
            # map target speed to VehicleControl for a carla vehicle
            veh_control = self.controller.generate_control(target_speed=target_speed,
                                                           next_waypoint=self._waypoint_buffer[0])
        else:
            # hold break
            veh_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            print('route is finished, please terminate the simulation manually.')

        # todo add debug attribute to set
        # print('throttle: ', veh_control.throttle)
        # print('steer: ', veh_control.steer)
        # print('brake: ', veh_control.brake)

        # original
        # init a vehicle control
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        control = veh_control

        return control























