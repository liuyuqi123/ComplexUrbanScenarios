"""
Use absolute kinetics for state representation.

Add as much as information into state representation.

"""

import numpy as np
import math

# kinetics method
from train.gym_carla.util_development.kinetics import angle_reg, get_distance_along_route

from train.gym_carla.modules.rl_modules.state_representation.state_manager_2 import StateManager


class StateManager4(StateManager):

    # todo add setter api to fix this value
    # maximum distance between ego and npc vehicle
    range_bound = 120.

    # threshold of front discrimination
    x_lower_limit = -10.0

    # compare different candidate state representation
    state_options = {
        # get as much information as possible, use absolute kinetics
        'absolute_all': {
            'ego_state_len': int(9),  # location_code(3) + (x, y, Vx, Vy, cos(yaw), sin(yaw)) + route_proportion(1)
            'npc_number': int(5),
            'npc_state_len': int(6),  # location and velocity in relative coordinate frame
        }
    }

    def __init__(self,
                 carla_api,
                 state_option='absolute_all',  # setting on ego state representation
                 attention=False,
                 # safe=False,
                 debug=False,
                 ):

        if state_option not in self.state_options.keys():
            raise ValueError('State option is not included, please check.')

        self.state_option = state_option

        super(StateManager, self).__init__(carla_api)

        # if using attention mechanism
        self.attention = attention
        # if using safety layer
        # self.safe

        # get state info
        self.ego_state_len = self.state_options[self.state_option]['ego_state_len']
        # npc vehicle number for state representation
        self.state_npc_number = self.state_options[self.state_option]['npc_number']
        # state of single npc vehicle
        self.npc_state_len = self.state_options[self.state_option]['npc_state_len']

        # dimension of original state vector
        if self.attention:
            self.state_len = self.ego_state_len + self.state_npc_number * self.npc_state_len + 1 + self.state_npc_number
        else:
            self.state_len = self.ego_state_len + self.state_npc_number * self.npc_state_len

        """
        # todo fix ob space initialization
        self.observation_space = None

        # fixme upper and lower bound is not correct
        # todo add method to fix the orders of state element
        # observation space for the gym.Env
        # ego + npc + attention(if using attention)
        low = np.array([float("-inf")] * self.state_len)
        high = np.array([float("inf")] * self.state_len)

        self.observation_space = Box(high=high, low=low, dtype=np.float32)
        """

        # ===============   self-defined attributes   ===============
        self.ego_vehicle = None
        # a dict to store kinetics of ego vehicle
        self.ego_info = {}
        # transform matrix from ego coord system to world system
        self.T_ego2world = None

        # list of npc vehicles(carla.Vehicle)
        self.npc_vehicles = []
        # state of current timestep
        self.state_array = None  # np.array

        # carla.Junction, if in a intersection scenario
        self.junction = None
        # the edge of junction bbox in coordinate
        self.junction_edges = {}

        # route of current episode, add a setter to get it
        self.ego_route = []
        self.start_waypoint = None
        self.end_waypoint = None  # not used for know
        self.route_length = None  # float

        # switch on visualization under debug mode
        self.debug = debug

    def set_ego_route(self, route, start_waypoint, end_waypoint):
        """
        A setter method.
        Get ego route from env.

        :param end_waypoint:
        :param start_waypoint:
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

    def check_is_front(self, veh_info):
        """
        Method for filter, check if a npc vehicle
        is in front of ego vehicle.

        This method will change veh_info dict.

        ps: Vehicle heading direction is fixed to x axis
        """
        flag = False

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

        if relative_loc[0] >= self.x_lower_limit:
            flag = True

        return flag

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
            if self.state_option == 'absolute_all':
                npc_state = self.get_single_state_absolute(veh_info)
            else:
                npc_state = self.get_single_state(veh_info)
            state += npc_state

        # check if need to padding state
        if _state_npc_number < self.state_npc_number:
            if self.debug:
                print('Only ', _state_npc_number, ' npc vehicles are suitable for state.')
                print(self.state_npc_number - _state_npc_number, 'padding states will be appended.')

            # todo fix this, get padding state from a class attribute, current is for absolute_all
            # padding state according to npc vehicles number
            padding_state = np.array([self.range_bound, self.range_bound, 0, 0, 1, 0])
            padding_state = list(padding_state)

            # padding state should be coordinate with single vehicle state
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

    def get_ego_state(self):
        """
        Get ego vehicle state.

        ego state for 'absolute_all' option:
         - location_code(3) + (x, y, Vx, Vy, cos(yaw), sin(yaw))

        """
        # check if junction is assigned
        if not self.junction:
            raise RuntimeError('Please assign junction for the state manager, using set_junction() method.')

        # todo add args to select different state representation
        if not self.state_option == 'absolute_all':  #
            raise RuntimeError('StateManger4 serves only absolute_all, please check your state option.')

        # update ego vehicle info
        self.ego_info = self.get_veh_info(self.ego_vehicle)

        # coordinate in global coordinate system
        location = self.ego_info['location']
        # get junction center location
        junction_center = self.junction.bounding_box.location
        relative_location = location - junction_center
        x_abs, y_abs = relative_location.x, relative_location.y

        # velocity in world coord system
        velocity = self.ego_info['velocity']
        v_x, v_y = velocity.x, velocity.y
        # ego speed
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # in m/s

        # vehicle heading direction
        rotation = self.ego_info['rotation']
        yaw = np.deg2rad(rotation.yaw)

        # abs location in world system
        x, y = location.x, location.y
        # edges of junction
        x_min = self.junction_edges['x_min']
        x_max = self.junction_edges['x_max']
        y_min = self.junction_edges['y_min']
        y_max = self.junction_edges['y_max']
        # todo fix condition check, using spawn point relative location
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

        # get distance proportion
        # check if ego route is
        # if not self.ego_route:
        #     raise RuntimeError('Ego route not found!')

        # remove driven distance
        # # distance ego vehicle has driven
        # driven_distance, _ = get_distance_along_route(self.map, self.ego_route, location)
        # # debug
        # if self.debug:
        #     if driven_distance < self.route_length:
        #         print('driven_distance: ', driven_distance)
        #     else:
        #         print('driven distance is probably wrong, please check.')
        # distance_proportion = driven_distance / self.route_length

        # reset state with ego state
        state = [x_abs, y_abs, v_x, v_y, np.cos(yaw), np.sin(yaw)] + position_code

        if self.debug:
            print('ego location(2D) to junction center: ', x_abs, y_abs)
            print('ego speed(km/h): ', speed)
            print('ego velocity(2D): ', v_x, v_y)

        return state

    def get_single_state_absolute(self, veh_info):
        """
        Get state of a single npc vehicle.

        Use absolute kinetics of the vehicle.
        """
        # check if junction is assigned
        if not self.junction:
            raise RuntimeError('Please assign junction for the state manager, using set_junction() method.')

        # location
        location = veh_info['location']
        junction_center = self.junction.bounding_box.location
        relative_location = location - junction_center
        x_abs, y_abs = relative_location.x, relative_location.y

        # velocity
        veh_velo = veh_info['velocity']
        V_x, V_y = veh_velo.x, veh_velo.y

        # heading direction
        rotation = veh_info['rotation']
        yaw = rotation.yaw
        yaw = np.deg2rad(yaw)

        npc_state = [x_abs, y_abs, V_x, V_y, np.cos(yaw), np.sin(yaw)]

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
