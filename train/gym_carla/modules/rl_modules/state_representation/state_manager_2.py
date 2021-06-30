"""
This class is manually reconstructed. We wish to merge more modules and APIs into it.

State representation module for gym_carla env.

Modified from original state_manager, major modification includes:

 - fix api to get state dimension
 - multiple ego state representation options
 - add args on using attention mechanism
 - add args on using multi-task frame
 - communication with CarlaEnv in multi-task training

todo
 - add safety constraint calculation and api
"""

import carla

import numpy as np
import math
import heapq

from gym.spaces import Box

# visualization module
from train.gym_carla.util_development.carla_color import *
from train.gym_carla.util_development.util_visualization import plot_actor_bbox, draw_2D_velocity

# kinetics method
from train.gym_carla.util_development.kinetics import angle_reg

from train.gym_carla.modules.carla_module import CarlaModule
# from gym_carla.envs.carla_module import CarlaRLModule


class StateManager(CarlaModule):
    """
    This class is responsible to manage state representation for a RL env.

    Major usages:
     - init the observation_space for the gym.Env
     - return a state vector for RL agent at each time step.

    Several state representation options:
     - with attention module:
        ego_state + npc_state + attention_mask
     - with multi-task framework:
        ego_state + npc_state + task_code
     - with both attention and multi-task module
        ego_state + npc_state + attention_mask + task_code

    In this class, we will try several different ego state representation.
    """

    # todo add setter api to fix this value
    # maximum distance between ego and npc vehicle
    range_bound = 100.

    # todo use args to set state representation parameters
    # dimension of state vector
    state_config = {
        'ego_state_len': int(4),  # location one-hot code, same as sumo experiments
        'npc_number': int(5),
        'npc_state_len': int(4),  # location and velocity in relative coordinate frame
    }

    def __init__(self,
                 carla_api,
                 attention=False,
                 # safe=False,
                 debug=False,
                 ):
        """
        Init state manager with experiment settings.
        """

        super(StateManager, self).__init__(carla_api)

        # if using attention mechanism
        self.attention = attention

        # todo if remove ob space init value
        self.observation_space = None

        # get state info
        self.ego_state_len = self.state_config['ego_state_len']
        self.state_npc_number = self.state_config['npc_number']  # npc vehicle number for state representation
        self.npc_state_len = self.state_config['npc_state_len']  # state of single npc vehicle

        # dimension of original state vector
        if self.attention:
            self.state_len = self.ego_state_len + self.state_npc_number * self.npc_state_len + 1 + self.state_npc_number
        else:
            self.state_len = self.ego_state_len + self.state_npc_number * self.npc_state_len

        # fixme upper and lower bound is not correct
        # todo add method to fix the orders of state element
        # observation space for the gym.Env
        # ego + npc + attention(if using attention)
        low = np.array([float("-inf")] * self.state_len)
        high = np.array([float("inf")] * self.state_len)

        self.observation_space = Box(high=high, low=low, dtype=np.float32)

        # ===============   attributes for RL   ===============
        self.ego_vehicle = None
        # a dict to store kinetics of ego vehicle
        self.ego_info = {}
        # transform matrix from ego coord system to world system
        self.T_world2ego = None
        self.T_ego2world = None

        # list of npc vehicles(carla.Vehicle)
        self.npc_vehicles = []
        # state of current timestep
        self.state_array = None  # np.array

        # carla.Junction, if in a intersection scenario
        self.junction = None
        # the edge of junction bbox in coordinate
        self.junction_edges = {}

        # switch on visualization under debug mode
        self.debug = debug

    def __call__(self, *args, **kwargs):
        """
        fixme finish this method to return state of current timestep

        :param args:
        :param kwargs:
        :return: state vector of current timestep
        """

        # state = []
        # print("state: ", state)
        # return state

        pass

    def get_junction_edges(self):
        """
        Parse junction info.
        """
        bbox = self.junction.bounding_box
        extent = bbox.extent
        location = bbox.location

        # todo fix for junction whose bbox has a rotation
        # rotation = bbox.rotation  # original rotation of the bbox

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

    def set_junction(self, junction):
        """
        Setter method, set junction of a intersection scenario.

        params: junction: A carla.Junction instance
        """
        self.junction = junction
        # get junction info after assigning junction
        self.get_junction_edges()

    def check_is_near(self, veh_info):
        """
        Method for filter, check if a vehicles is near ego.
        """
        # condition result
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
        Method for filter, check if a npc vehicle
        is in front of ego vehicle.

        This method will change veh_info dict.

        ps: Vehicle heading direction is fixed to x axis
        """
        flag = False

        # todo this param must be reset to a larger value in right turn task
        # threshold of front discrimination
        dist_threshold = -30.0

        # ego information
        ego_location = self.ego_info['location']
        # ego_x, ego_y = ego_location.x, ego_location.y
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
        # update transform matrix
        trans_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [-1*np.sin(ego_yaw), np.cos(ego_yaw)]])
        self.T_world2ego = trans_matrix
        self.T_ego2world = trans_matrix.T

        # relative location vector in ego coord frame
        relative_loc = np.dot(trans_matrix, relative_loc)
        relative_yaw = yaw - ego_yaw  # npc relative to ego

        # regularize angle
        relative_yaw = angle_reg(relative_yaw)

        # add relative location and velocity into veh_info dict
        veh_info['relative_location'] = relative_loc  # 2D relative location in ego coord system
        veh_info['relative_yaw'] = relative_yaw

        if relative_loc[0] >= dist_threshold:
            flag = True

        return flag

    def filter_npc_vehicles(self):
        """
        This method is responsible for filtering suitable
        npc vehicles for state representation.

        Return a dict contains info of candidate npc vehicles.

        Currently, 2 rules are considered:
         - if npc vehicle in range bound
         - if npc vehicle is in front of ego vehicle
        """
        # list for carla.Vehicles
        near_npc_vehicles = []

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

        # select npc vehicles and info for state representation
        selected_npc_vehicles = heapq.nsmallest(self.state_npc_number,  # int, number of npc vehicles
                                                candidate_npc_vehicles,  # list in which stores the dict
                                                key=lambda s: s['distance'])

        return selected_npc_vehicles

    def update_vehicles(self):
        """
        Update vehicle list of current timestep.

        This method is supposed to be called each timestep
        """
        self.npc_vehicles = []
        self.ego_vehicle = None

        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # carla.Actorlist, iterable

        if vehicle_list:
            for veh in vehicle_list:  # veh is carla.Vehicle
                attr = veh.attributes  # dict
                # filter ego vehicle
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':  # name hero is responsible for physics
                    self.ego_vehicle = veh
                else:
                    self.npc_vehicles.append(veh)

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
        relative_velocity = np.dot(self.T_world2ego, _rel_velo)

        # in km/h
        relative_velocity = relative_velocity * 3.6

        # current state representation assumption:
        # only consider relative location and velocity
        relative_location = veh_info['relative_location']

        # todo add different state representation option
        # add vehicle rotation info

        npc_state = np.concatenate((relative_location, relative_velocity), axis=0)
        # in list
        npc_state = list(npc_state)

        return npc_state

    def get_ego_state(self):
        """
        Get ego vehicle state.
        """
        # update ego vehicle info
        self.ego_info = self.get_veh_info(self.ego_vehicle)

        # coords
        location = self.ego_info['location']
        x, y = location.x, location.y
        # speed
        velocity = self.ego_info['velocity']
        speed = self.vel2speed(velocity)

        state = [speed]

        # todo add args to select state representation option
        if self.ego_state_len == int(4):  # using position_code
            if not self.junction:
                raise RuntimeError('Please assign junction for the state manager, using set_junction() method.')
            # edges of junction
            x_min = self.junction_edges['x_min']
            x_max = self.junction_edges['x_max']
            y_min = self.junction_edges['y_min']
            y_max = self.junction_edges['y_max']

            # position_code refer to which part of route ego is in
            # todo add args for ego vehicle being in random intersection
            before_junction = y >= y_max
            in_junction = x_min <= x <= x_max and y_min <= y <= y_max

            if in_junction:
                position_code = [0, 1, 0]
            else:
                if before_junction:
                    position_code = [1, 0, 0]
                else:  # after leaving junction
                    position_code = [0, 0, 1]

            state = position_code + state

        # another state representation option
        elif self.ego_state_len == int(3):

            state = [x, y] + state

        if self.debug:
            print('ego location(2D): ', x, y)
            print('ego speed(km/h): ', speed)
            # plot ego vehicle
            plot_actor_bbox(self.ego_vehicle)

        return state

    def get_attention_mask(self, selected_npc_vehicles):
        """
        Get attention mask according to current selected vehicles number.
        """
        # musk size equals to vehicle number(npc number + ego)
        total_veh_num = self.state_npc_number + 1
        mask_size = int(total_veh_num)

        mask = list(np.ones(mask_size))

        # if not enough npc vehicles, padding state vector
        if len(selected_npc_vehicles) < self.state_npc_number:

            # missing npc number
            missing_npc_num = self.state_npc_number - len(selected_npc_vehicles)

            # pop missing bit
            for i in range(missing_npc_num):
                mask.pop()

            # append 0 to missing bit
            for i in range(missing_npc_num):
                mask.append(0)

        return mask

    def get_state(self):
        """
        Get state vector with attention.

        Get state of current timestep for RL module.

        This method is re-writed for attention mechanism.
        """
        # update active vehicles
        self.update_vehicles()

        # init state list with ego state
        if self.ego_vehicle:  # check ego vehicle
            state = self.get_ego_state()
        else:
            raise RuntimeError('Ego vehicle is not found!')

        # filter npc vehicles
        selected_npc_vehicles = self.filter_npc_vehicles()  # in dict
        # the real npc vehicle number for state representation
        _state_npc_number = len(selected_npc_vehicles)

        # visualize npc vehicles for state
        if self.debug:
            # get actors from the dict
            plot_veh_list = [info_dict['vehicle'] for info_dict in selected_npc_vehicles]
            self.visualize_npc_vehicles(plot_veh_list)

        # append npc vehicle state
        for veh_info in selected_npc_vehicles:
            npc_state = self.get_single_state(veh_info)
            state += npc_state

        # check if need to padding state
        if _state_npc_number < self.state_npc_number:
            if self.debug:
                print('Only ', _state_npc_number, ' npc vehicles are suitable for state.')
                print(self.state_npc_number - _state_npc_number, 'padding states will be appended.')

            # todo add padding_state into class attribute
            # padding state according to npc vehicles number
            padding_state = np.array([self.range_bound, self.range_bound, 0, 0])
            padding_state = list(padding_state)
            # todo padding state should be coordinate with single vehicle state
            if len(padding_state) is not self.npc_state_len:
                raise RuntimeError('Padding npc state is different from definition, please check!')

            for _ in range(int(self.state_npc_number - _state_npc_number)):
                state += padding_state

        # ================  add attention mask   ================
        if self.attention:
            mask = self.get_attention_mask(selected_npc_vehicles)
            # todo add args to check attention state length
            state = state + mask

            if self.debug:
                print('Using attention mechanism. The state length is: ', len(state))

        # check state dimension
        if len(state) is not self.state_len:
            raise RuntimeError('state length is not coordinate, please check.')

        # store state vector in ndarray format
        state_array = np.array(state)
        self.state_array = state_array

        return state_array

    # =====================  staticmethod  =====================
    @staticmethod
    def visualize_npc_vehicles(vehicle_list):
        """
        Visualize vehicles in vehicle_list
        """
        for veh in vehicle_list:
            # plot bbox of vehicles
            plot_actor_bbox(veh, color=magenta)
            # draw velocity vector
            draw_2D_velocity(veh, life_time=0.1)

            # todo visualize relative location by
            #  drawing relative location vector

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

