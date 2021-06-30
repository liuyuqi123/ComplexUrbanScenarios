"""
Modifications on state representation sumo.

state option is sumo_1
"""

import numpy as np
import math
import heapq

from gym.spaces import Box

# visualization module
from train.gym_carla.util_development.util_visualization import plot_actor_bbox
from train.gym_carla.util_development.kinetics import get_distance_along_route, get_inverse_transform_matrix

# from gym_carla.envs.carla_module import CarlaRLModule

from train.gym_carla.modules.rl_modules.state_representation.state_manager_2 import StateManager


class StateManager5(StateManager):

    state_config = {
        'ego_state_len': int(4),  # default
        'npc_number': int(5),
        'npc_state_len': int(6),  # location and velocity in relative coordinate frame
    }

    def __init__(self,
                 carla_api,
                 attention=False,
                 debug=False,
                 ):

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

        # list of all npc vehicles(carla.Vehicle)
        self.npc_vehicles = []
        # NPC vehicles for state representation, dict: (carla.Vehicle, info_dict)
        self.state_npc_vehicles = {}

        # state of current timestep
        self.state_array = None  # np.array

        # carla.Junction, if in a intersection scenario
        self.junction = None
        # the edge of junction bbox in coordinate
        self.junction_edges = {}

        # switch on visualization under debug mode
        self.debug = debug

    def set_ego_route(self, route, start_waypoint, end_waypoint):
        """
        A setter method.
        Get ego route from env.

        :param route: list of tuple (transform, RoadOption)
        """
        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint

        # short distance to avoid get_distance_along_route error
        _dist = 2.0
        pre_wp = start_waypoint.previous(_dist)[0]
        after_ap = end_waypoint.next(_dist)[0]

        # set and extend route
        road_option = route[0][1]
        self.ego_route = [(pre_wp.transform, road_option)] + route + [(after_ap.transform, road_option)]

        # get total length of the route
        end_location = self.ego_route[-1][0].location
        # if using route length
        self.route_length, _ = get_distance_along_route(self.map, self.ego_route, end_location)

    def get_ego_state(self):
        """
        Get ego vehicle state.
        """
        # update ego vehicle info
        self.ego_info = self.get_veh_info(self.ego_vehicle)
        # coordinate in global coordinate system
        location = self.ego_info['location']
        x, y = location.x, location.y
        # speed velocity in world coord system
        velocity = self.ego_info['velocity']
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # in m/s
        # init state with ego speed
        state = [speed]

        if not self.junction:
            raise RuntimeError('Please assign junction for the state manager, using set_junction() method.')
        # edges of junction
        x_min = self.junction_edges['x_min']
        x_max = self.junction_edges['x_max']
        y_min = self.junction_edges['y_min']
        y_max = self.junction_edges['y_max']
        # position_code refer to which part of route ego is at
        before_junction = y >= y_max
        in_junction = x_min <= x <= x_max and y_min <= y <= y_max
        # get position code
        if in_junction:
            position_code = [0, 1, 0]
        else:
            if before_junction:
                position_code = [1, 0, 0]
            else:  # after leaving junction
                position_code = [0, 0, 1]

        position_code = list(map(float, position_code))
        state = position_code + state

        if self.debug:
            # print('ego location(2D): ', x, y)
            # print('ego speed(km/h): ', speed)

            # plot ego vehicle
            plot_actor_bbox(self.ego_vehicle)

        return state

    def get_single_state(self, veh_info):
        """
        Get state of a single npc vehicle.

        todo consider npc vehicle rotation
        """
        # calculate relative velocity
        ego_velo = self.ego_info['velocity']
        ego_yaw = self.ego_info['rotation'].yaw
        veh_velo = veh_info['velocity']
        rel_velo = veh_velo - ego_velo
        # use ndarray
        _rel_velo = np.array([rel_velo.x, rel_velo.y])
        # 2D velocity in ego coord frame
        relative_velocity = np.dot(self.T_world2ego, _rel_velo)  # in m/s

        # get from veh_info dict
        relative_location = veh_info['relative_location']
        npc_state = np.concatenate((relative_location, relative_velocity), axis=0)
        npc_state = list(npc_state)  # in list

        # get relative yaw
        rotation = veh_info['rotation']
        yaw = rotation.yaw
        relative_yaw = yaw - ego_yaw
        relative_yaw = np.deg2rad(relative_yaw)
        heading_direction = [np.cos(relative_yaw), np.sin(relative_yaw)]

        npc_state += heading_direction

        return npc_state

    def get_ego_vehicle(self):
        """
        A Getter to get ego vehicle from instance.

        This method serves the test_agent for scenario test.
        """
        if self.ego_vehicle:
            return self.ego_vehicle
        else:
            raise RuntimeError('Ego vehicle not found!')

    def get_state(self):
        """
        Get state vector with attention.

        Get state of current timestep for RL module.

        This method is re-writed for attention mechanism.
        """
        # update alive vehicles
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

        # todo need to test this line
        # store state NPC vehicles
        self.state_npc_vehicles = {}
        for dic in selected_npc_vehicles:
            self.state_npc_vehicles[dic['vehicle']] = dic

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
            padding_state = np.array([self.range_bound, self.range_bound, 0, 0, 1, 0])
            padding_state = list(padding_state)
            # todo padding state should be coordinate with single vehicle state
            if len(padding_state) is not self.npc_state_len:
                raise RuntimeError('Padding npc state is different from definition, please check!')

            for _ in range(int(self.state_npc_number - _state_npc_number)):
                state += padding_state

        # todo add attention mechanism, need to fix API of the carla env
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

    def get_ttc(self):
        """
        Calculate time to collision through state vector(StateManager5).
        :return:
        """
        ttc_list = []

        for i in range(self.state_config['npc_number']):
            index = 4 + i*6
            clipped_state = self.state_array[index:index+4]

            # relative location
            loc_vector = np.array([
                clipped_state[0],
                clipped_state[1],
            ])
            norm_loc_vector = np.linalg.norm(loc_vector)

            # relative distance
            rel_distance = np.linalg.norm(loc_vector) - 2.67 * 2  # 2.67 is collision radius of mkz, in meters
            rel_distance = np.clip(rel_distance, 0.1, 9999)

            # relative velocity
            vel_vector = np.array([
                clipped_state[2],
                clipped_state[3],
            ])

            # velocity vector projected on relative location vector direction
            vel_projection = np.dot(loc_vector, vel_vector) / norm_loc_vector  # minus value
            vel_projection = np.clip(vel_projection, -1 * np.Inf, -1e-8)

            ttc = rel_distance / (-1 * vel_projection)
            ttc_list.append(ttc)

            # # print ttc value
            # print('TTC of vehicle {}: {}'.format(int(i), ttc))

        # get the minimum
        ttc = min(ttc_list)

        return ttc

    def get_precise_ttc(self):
        """
        This is a modified version of ttc calculation.
        Shape of vehicle will be considered.

        The distance gap will be calculated using distance of center minus vehicle overlap distance.

        :return:
        """

        #
        # get transform matrix, world2vehicle
        transform_ego = self.ego_info['transform']
        trans_mat_ego = get_inverse_transform_matrix(transform_ego)

        # coordinate
        location_ego = self.ego_info['location']
        x_ego, y_ego = location_ego.x, location_ego.y
        yaw_ego = transform_ego.yaw

        # todo improve this part, get the bbox info for once
        #　ego vehicle bbox projection length
        bbox_ego = self.ego_vehicle.bounding_box
        # theta_0_ego = np.arctan(bbox_ego.extent.y, bbox_ego.extent.x)  # in rads

        # 2D relative velocity vector
        velocity_ego = self.ego_info['velocity']

        # store ttc value of each vehicle
        ttc_list = []

        # todo check if the dict is empty
        for vehicle, info_dict in self.state_npc_vehicles.items():
            transform_npc = get_inverse_transform_matrix(info_dict['transform'])
            trans_mat_npc = get_inverse_transform_matrix(transform_npc)

            # coordinate
            location_npc = self.ego_info['location']
            x_npc, y_npc = location_npc.x, location_npc.y
            yaw_npc = transform_npc.yaw

            # relative location is stored in info_dict
            # relative_location = info_dict['relative_location']  # 2D relative location in ego coord system
            relative_yaw = info_dict['relative_yaw']

            # todo check if calculation is correct, compare with info_dict value
            # 3D relative location in world coord system
            # ego related to npc
            relative_location = np.array([x_ego - x_npc, y_ego - y_npc, 0.])

            # todo check this value
            # distance of the centers of the vehicles
            center_distance = np.linalg.norm(relative_location)

            # NPC vehicle may use different bp from ego vehicle
            # bbox projection length
            bbox_npc = vehicle.bounding_box
            # theta_0_npc = np.arctan(bbox_npc.extent.y, bbox_npc.extent.x)  # in rads

            # 3D relative location vector
            relative_loc_vector_npc2ego = np.dot(trans_mat_ego, -1 * relative_location.T)  # in ego coord system
            relative_loc_vector_ego2npc = np.dot(trans_mat_ego, relative_location.T)  # in npc coord system

            # calculate current theta
            # theta_ego = np.arctan(relative_loc_vector_npc2ego[0], relative_loc_vector_npc2ego[1])
            # theta_npc = np.arctan(relative_loc_vector_ego2npc[0], relative_loc_vector_ego2npc[1])

            # todo debug this part
            def get_bbox_dist(location_vector, bbox):
                """
                Get current distance gap inside the bbox.

                :param bbox: bounding box of the target vehicle
                :param location_vector: relative location vector in coord system of current vehicle
                :return: distance inside the bbox
                """
                # this is current theta value of relative location vector
                _theta = np.arctan(location_vector[0], location_vector[1])

                # # todo fix and check this the vector in 3rd and 4th Quadrant will not effect bbox distance
                # # check if vector is in 3rd and 4th Quadrant
                # if location_vector[0] <= 0.:
                #     if -1*_theta <= theta <= _theta:
                #     _theta = 0.5 * np.pi - _theta
                #     l_e = bbox.x / np.cos(theta_e)
                # else:
                #     l_e = bbox.y / np.sin(theta_e)

                # in rads
                theta_0 = np.arctan(bbox_ego.extent.y, bbox_ego.extent.x)

                if -1*theta_0 <= _theta <= theta_0:
                    l_e = bbox.x / np.cos(_theta)
                else:
                    l_e = bbox.y / np.sin(_theta)

                return l_e

            # distance in ego bbox
            bbox_dist_ego = get_bbox_dist(
                    location_vector=relative_loc_vector_npc2ego,
                    bbox=bbox_ego,
            )

            # distance in npc bbox
            bbox_dist_npc = get_bbox_dist(
                location_vector=relative_loc_vector_ego2npc,
                bbox=bbox_npc,
            )

            # center distance
            center_distance = location_ego.distance(location_npc)
            # net distance
            net_distance = center_distance - bbox_dist_ego - bbox_dist_npc

            # =====  get velocity projection for ttc calculation  =====
            # get relative velocity vector
            velocity_npc = info_dict['velocity']  # carla.Vector3D

            # carla.Vector3D __sub__ method
            # this velocity vector is in world coord system
            relative_velocity = velocity_npc - velocity_ego
            # to ndarray
            relative_velocity = np.array([relative_velocity.x, relative_velocity.y, relative_velocity.z])

            # transform to ego coordinate system, 3D
            relative_velocity = np.dot(trans_mat_ego, relative_velocity)

            # norm velocity
            norm_relative_velocity = np.linalg.norm(relative_velocity)

            # normalized relative location vector
            # todo check this value, should be equal to center_distance
            _rel_loc_vector = relative_loc_vector_npc2ego / np.linalg.norm(relative_loc_vector_npc2ego)

            # velocity projection value
            vel_projection = np.dot(_rel_loc_vector, relative_velocity)  # minus value

            # clip to a minus value
            vel_projection = np.clip(vel_projection, -1 * np.Inf, -1e-8)

            # calculate ttc value
            ttc = net_distance / (-1 * vel_projection)

            # element is the dict
            ttc_list.append(
                {
                    vehicle: ttc,
                }
            )

            # # debug to print ttc value
            # print('TTC of vehicle {}: {}'.format(int(i), ttc))

        # todo filter the min ttc of all npc vehicles
        # get the minimum ttc value
        ttc = heapq.nsmallest(
            1,  # int, number of npc vehicles
            ttc_list,  # list in which stores the dict
            key=lambda s: s['vehicle']
        )

        return ttc

    # # todo get a precise ttc, consider vehicle bbox rotation
    # def get_contraint_value(self):
    #     """"""
    #     # get ttc related info from state manager
    #
    #     # info of ego vehicle and the nearest vehicle
    #     x_e, y_e, yaw_e, transform_e =
    #     x_n, y_n, yaw_n, transform_n =
    #
    #     # get transform matrix
    #     # world2vehicle
    #     T_e =
    #     T_n =
    #
    #     # get relative location vector
    #     _vec = np.array([[x_n-x_e], [y_n-y_e], [0.]])
    #     vec_n2e = np.dot(T_e, _vec)
    #     vec_e2n = np.dot(T_e, -1*_vec)  # todo test multiply method
    #
    #     # shifting angle of the bbox
    #     bbox =
    #     theta_0 = np.arctan(bbox.y, bbox.x)  # todo check red
    #
    #     # calculate in bbox length
    #     theta_e = np.arctan(vec_n2e[0], vec_n2e[1])
    #     theta_n = np.arctan(vec_e2n[0], vec_e2n[1])
    #
    #     if theta_e <= theta_0:
    #         l_e = bbox.x / np.cos(theta_e)
    #     else:
    #         l_e = bbox.y / np.sin(theta_e)
    #
    #     if theta_e <= theta_0:
    #         l_n = bbox.x / np.cos(theta_n)
    #     else:
    #         l_n = bbox.y / np.sin(theta_n)
    #
    #     # distance between center
    #     dist =
    #
    #     net_distance = dist - l_n - l_e
    #
    #     return net_distance


