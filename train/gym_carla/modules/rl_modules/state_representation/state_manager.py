"""
State representation for RL in carla

This state module is same as sumo junction turning experiments.


"""

import numpy as np
import math
import heapq

import carla

from train.gym_carla.util_development.util_visualization import plot_actor_bbox, draw_2D_velocity
from train.gym_carla.util_development.carla_color import *

from train.gym_carla.modules.carla_module import CarlaModule
# from gym_carla.envs.carla_module import CarlaRLModule

# util methods
from train.gym_carla.util_development.kinetics import angle_reg

from gym.spaces import Box


# an example of rl_config
# ref on rl_config file
default_rl_config = {
    'ego_feature_num': int(4),  # with a code respond to current location, for attention query
    'npc_number': int(5),
    'npc_feature_num': int(4),
}


# todo in developing
class CarlaRLModule(CarlaModule):
    """
    This class is responsible for managing a rl module for carla experiments.

    todo add optional args to set if using attention and multi-task code

    """

    def __init__(self, carla_api, rl_config=None):
        """

        todo use state and action as example

        todo rl api is supposed to be developed

        :param carla_api:
        :param rl_config:
        """
        # init basic module class
        super(CarlaRLModule, self).__init__(carla_api)

        # this class should

    def __call__(self, *args, **kwargs):
        pass


class StateManager(CarlaModule):
    """
    This class is responsible to manage state representation for RL agent in carla experiments.

    todo check if using a RL module as a parent class

    state is represented as:

    ego_state + npc_state + task code


    Major usages:
     - init a state space for the gym.Env
     - extract state representation for RL agent at each time step.

    """

    # todo add an api to set the state config
    # dimension of state vector
    state_config = {
        'ego_state_len': int(3),
        'npc_number': int(5),
        'npc_state_len': int(4),
    }

    def __init__(self, carla_api):
        """
        Init the state manager.

        param rl_config: a dict contains rl agent config parameters.
        """

        super(StateManager, self).__init__(carla_api)

        # self.config = rl_config

        # todo if remove ob space init value
        self.observation_space = None

        # todo add input args
        self.range_bound = 100.  # maximum distance between ego and npc vehicle

        # get state info
        self.ego_state_len = self.state_config['ego_state_len']
        self.state_npc_number = self.state_config['npc_number']  # npc vehicle number for state representation
        self.npc_state_len = self.state_config['npc_state_len']  # state of single npc vehicle

        # dim of state vector
        self.state_len = self.ego_state_len + self.state_npc_number * self.npc_state_len

        low = np.array([float("-inf")] * self.state_len)
        high = np.array([float("inf")] * self.state_len)

        self.observation_space = Box(high=high, low=low, dtype=np.float32)

        # ego vehicle carla.Vehicle
        self.ego_vehicle = None
        # a dict stores kinetics of ego vehicle
        self.ego_info = {}

        # transform matrix from ego coord system to world system
        self.T_ego2world = None

        # list to store npc vehicles(carla.Vehicle)
        self.npc_vehicles = []

        # state of each timestep
        self.state_array = None  # ndarray

        # visualization option
        self.debug = True

        # todo fix this attribute with a general api
        self.junction = None
        self.junction_edges = None

    def init_ob_space(self):
        """
        todo this method may not be useful

        Init the observation space for RL.

        todo check upper and lower bound of state space
        """

        # todo state length is supposed to be in config file

        # extract config parameters from the config dict
        # low = np.array([float("-inf")] * self.config['state_len'])
        # high = np.array([float("-inf")] * self.config['state_len'])

        low = np.array([float("-inf")] * self.state_len)
        high = np.array([float("-inf")] * self.state_len)

        self.observation_space = Box(high=high, low=low, dtype=np.float32)

    @staticmethod
    def vel2speed(vel):
        """
        Get speed from velocity, in km/s.

        vel is in m/s from carla.
        """
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    @staticmethod
    def get_veh_info(vehicle: carla.Vehicle):
        """
        Get info dict of a vehicle.
        :return:
        """
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        velocity = vehicle.get_velocity()
        bounding_box = vehicle.bounding_box

        info_dict = {
            'vehicle': vehicle,
            'transform': transform,
            'location': location,
            'rotation': rotation,
            'bounding_box': bounding_box,
            'velocity': velocity,
        }

        return info_dict

    def get_ego_state(self):
        """
        Get ego vehicle state.
        """
        # update ego vehicle info
        self.ego_info = self.get_veh_info(self.ego_vehicle)

        location = self.ego_info['location']
        x, y = location.x, location.y

        velocity = self.ego_info['velocity']
        speed = self.vel2speed(velocity)

        # todo fix ego state of x, y coords
        # ego state vector
        ego_state = [x, y, speed]

        if self.debug:
            print('ego location(2D): ', x, y)
            print('ego speed(km/h): ', speed)

            # plot ego vehicle in server
            # todo fix this method, plot_actor
            plot_actor_bbox(self.ego_vehicle)

        return ego_state

    def set_junction(self, junction):
        """
        Get current junction in turning scenario.

        params: junction: A carla.Junction instancec
        """
        self.junction = junction
        # get junction info after assigning junction
        self.get_junction_info()

    def get_junction_info(self):
        """
        Parse junction info.
        """

        bbox = self.junction.bounding_box
        extent = bbox.extent
        location = bbox.location
        # todo check if junction bbox has a rotation
        rotation = bbox.rotation  # original rotation of the bbox

        # get junction vertices
        x_max = location.x + extent.x
        x_min = location.x - extent.x
        y_max = location.y + extent.y
        y_min = location.y - extent.y

        junction_edges = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
        }

        self.junction_edges = junction_edges

    def get_ego_state2(self):
        """
        Add position flag into ego state.

        position flag is a one-hot code to identify the period of the route.
        """
        # update ego vehicle info
        self.ego_info = self.get_veh_info(self.ego_vehicle)

        location = self.ego_info['location']
        x, y = location.x, location.y

        velocity = self.ego_info['velocity']
        speed = self.vel2speed(velocity)

        # todo fix when ego vehicle is in different route
        x_min = self.junction_edges['x_min']
        y_max = self.junction_edges['y_max']

        # get the position flag of ego vehicle
        if y <= y_max:
            if x >= x_min:  # in the junction
                position_flag = [0, 1, 0]
            else:  # after leaving the junction
                position_flag = [0, 0, 1]
        else:  # before get into junction
            position_flag = [1, 0, 0]

        ego_state = position_flag + [speed]

        if self.debug:
            print('ego location(2D): ', x, y)
            print('ego speed(km/h): ', speed)

            # plot ego vehicle in server
            # todo fix this method, plot_actor
            plot_actor_bbox(self.ego_vehicle)

        return ego_state

    def update_vehicle_list(self):
        """
        Update vehicle list of current timestep.

        This method is supposed to be called each timestep

        todo this method may be generally used in many modules, should be ticked each timestep
        """
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # carla.Actorlist, iterable
        self.npc_vehicles = []
        self.ego_vehicle = None

        if vehicle_list:
            for veh in vehicle_list:  # veh is carla.Vehicle
                attr = veh.attributes  # dict
                # filter ego vehicle
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':  # name hero is responsible for physics
                    self.ego_vehicle = veh
                else:
                    self.npc_vehicles.append(veh)

        # todo deal with if ego vehicle is not found
        if not self.ego_vehicle:
            raise RuntimeError('ego vehicle missing!')

    def check_is_near(self, veh_info):
        """
        Check if vehicle is near ego vehicle.

        Class attribute range_bound
        """
        flag = False

        location = veh_info['location']
        ego_location = self.ego_info['location']
        distance = ego_location.distance(location)  # in meters

        # add distance into veh_info dict
        veh_info['distance'] = distance

        # check if this vehicle is within the range bound
        if distance <= self.range_bound:
            flag = True

        return flag

    def check_is_front(self, veh_info):
        """
        Check if a npc vehicle is in front of ego vehicle.

        Vehicle heading direction is fixed to x axis
        """
        flag = False

        # threshold of front discrimination
        dist_threshold = 5.0

        # ego info
        ego_location = self.ego_info['location']
        ego_x, ego_y = ego_location.x, ego_location.y
        ego_rotation = self.ego_info['rotation']
        ego_yaw = np.radians(ego_rotation.yaw)

        # check if this npc is in front of ego vehicle
        location = veh_info['location']
        rotation = veh_info['rotation']
        yaw = np.radians(rotation.yaw)

        # relative location in global coordinate system
        relative_location = location - ego_location
        relative_loc = np.array([relative_location.x, relative_location.y])

        # todo check rads or degree
        # transform matrix
        trans_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [-1*np.sin(ego_yaw), np.cos(ego_yaw)]])
        # update transform matrix
        self.T_ego2world = trans_matrix

        # relative location vector in ego coord frame
        relative_loc = np.dot(trans_matrix, relative_loc)
        relative_yaw = yaw - ego_yaw  # npc relative to ego

        # regularize angle
        relative_yaw = angle_reg(relative_yaw)

        # add relative location and velocity into veh_info dict
        veh_info['relative_location'] = relative_loc  # 2D
        veh_info['relative_yaw'] = relative_yaw

        if relative_loc[0] >= -1 * dist_threshold:
            flag = True

        return flag

    def get_single_state(self, veh_info):
        """
        Get state of a single npc vehicle.
        """
        # calculate relative velocity
        ego_velo = self.ego_info['velocity']
        veh_velo = veh_info['velocity']
        rel_velo = veh_velo - ego_velo
        # use ndarray
        _rel_velo = np.array([rel_velo.x, rel_velo.y])
        # 2D velocity in ego coord frame
        relative_velocity = np.dot(self.T_ego2world, _rel_velo)

        # todo 2 different state representation option
        # option 1: consider same settings as sumo
        # assume that vehicle velocity direction is same as heading direction
        #

        # option 2: consider relative location and velocity separately
        relative_location = veh_info['relative_location']

        npc_state = np.concatenate((relative_location, relative_velocity), axis=0)

        npc_state = list(npc_state)  # return in list

        return npc_state

    @staticmethod
    def compare_yaw_angle(vehicle):
        """
        Compare angle difference between vehicle heading and velocity

        Yaw angle refers to heading direction of a vehicle,
        compares with direction of velocity vector.

        :return: angle difference
        """
        #
        transform = vehicle.get_transform()
        rotation = transform.rotation
        yaw = np.radians(rotation.yaw)
        heading_direction = np.array([np.cos(yaw), np.sin(yaw)])

        velocity = vehicle.get_velocity()
        velo_2D = np.array([velocity.x, velocity.y])

        cos_angle = np.dot(heading_direction, velo_2D) / np.linalg.norm(heading_direction) / np.linalg.norm(velo_2D)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)  # in radians
        angle = np.degrees(angle)

        return angle

    def filter_npc_vehicles(self):
        """
        Filter npc vehicles for state representation.

        Return a dict contains info of potential npc vehicles.

        2 conditions are considerd:
         - if npc vehicle in range bound
         - if npc vehicle is in front of ego vehicle
        """
        near_npc_vehicles = []  # carla.Vehicles

        for index, veh in enumerate(self.npc_vehicles):
            veh_info = self.get_veh_info(veh)  # return a info dict
            # add distance into veh_info
            is_near = self.check_is_near(veh_info)

            if is_near:
                near_npc_vehicles.append(veh_info)

        # select vehicles in front of ego vehicle
        candidate_npc_vehicles = []
        for veh_info in near_npc_vehicles:
            is_front = self.check_is_front(veh_info)
            if is_front:
                candidate_npc_vehicles.append(veh_info)

        # select npc vehicles for state representation
        selected_npc_vehicles = heapq.nsmallest(self.state_npc_number,  # int, number of npc vehicles
                                                candidate_npc_vehicles,  # list in which stores the dict
                                                key=lambda s: s['distance'])

        return selected_npc_vehicles

    def get_state(self):
        """
        Get state of current timestep for RL module.
        """
        # update active vehicles
        self.update_vehicle_list()

        # check ego vehicle
        if self.ego_vehicle.is_alive:
            # state = self.get_ego_state()

            state = self.get_ego_state2()
        else:
            raise Exception("Ego not found!")

        # filter npc vehicles
        selected_npc_vehicles = self.filter_npc_vehicles()

        # visualize npc vehicles for state
        if self.debug:
            # get actors from the dict
            vis_veh_list = [info_dict['vehicle'] for info_dict in selected_npc_vehicles]
            self.visualize_npc_vehicles(vis_veh_list)

        for veh_info in selected_npc_vehicles:
            npc_state = self.get_single_state(veh_info)
            state += npc_state

        # padding state vector
        if len(selected_npc_vehicles) < self.state_npc_number:
            if self.debug:
                print(len(selected_npc_vehicles), ' npc vehicles are suitable for state.')

            # todo padding state should be coordinate with single vehicle state
            padding_state = np.array([self.range_bound, self.range_bound, 0, 0])
            padding_state = list(padding_state)

            # desired state length
            # todo get this from args
            state_len = 23

            while len(state) < state_len:
                state += padding_state

        # ==================================================
        """

        # todo add use attention option api, add it into the config file as well
        # if use_attention:
        # mask = list(np.ones(self.mask_size))
        # if len(state) < self.state_size:
        #     zero_padding_num = int((self.state_size - len(state)) / self.npc_feature_num)
        #     for _ in range(zero_padding_num):
        #         mask.pop()
        #     for _ in range(zero_padding_num):
        #         mask.append(0)
        #     while len(state) < self.state_size:
        #         state.append(0)
        # state_mask = state + mask
        # state_mask = np.array(state_mask)

        # if not using attention
        # mask = list(np.ones(self.mask_size))
        if len(state) < self.state_size:
            # zero_padding_num = int((self.state_size - len(state)) / self.npc_feature_num)
            # for _ in range(zero_padding_num):
            #     mask.pop()
            # for _ in range(zero_padding_num):
            #     mask.append(0)
            while len(state) < self.state_size:
                state.append(0)
        # state_mask = state + mask
        # state_mask = np.array(state_mask)
        
        """

        # ==================================================

        # ndarray format
        state_array = np.array(state)
        self.state_array = state_array

        return state_array

    def visualize_npc_vehicles(self, vehicle_list):
        """
        Visualize all npc vehicles.
        """
        # visualize vehicles for state representation
        for veh in vehicle_list:
            # plot bbox
            plot_actor_bbox(veh, color=magenta)
            # draw velocity vector
            draw_2D_velocity(veh, life_time=0.1)

        # todo how to visualize relative location??

        # print('All vehicles are plot in world.')

    def get_ego_vehicle(self):
        """
        A Getter to get ego vehicle from instance.
        """
        if self.ego_vehicle:
            return self.ego_vehicle
        else:
            raise RuntimeError('Ego vehicle not found!')

    def __call__(self, *args, **kwargs):
        """
        When the instance of state manager is called, the newest state tensor is supposed to be returned.

        todo add other methods, update npc list..

        :param args:
        :param kwargs:
        :return:
        """

        state = self.get_state()

        # print("state is updated: ", state)
        return state