"""
A developing version based on state_manager_2.

Inherit class <state_manager_2> and rewrite some methods.
"""

import numpy as np
import math

# visualization module
from train.gym_carla.util_development.util_visualization import plot_actor_bbox
from train.gym_carla.util_development.kinetics import get_distance_along_route

# kinetics method

# from gym_carla.envs.carla_module import CarlaRLModule

from train.gym_carla.modules.rl_modules.state_representation.state_manager_2 import StateManager


# compare different ego state representation
ego_state_options = {
    'sumo': int(4),  # [speed, position_code(3)]
    'kinetics': int(3),  # [x, y, speed], coordinates relative to junction center
    'driven_dist': int(2),  # [driven_distance_proportion, speed]
    # 'hybrid': int(6),  # [x, y, speed, position_code(3)] todo check this

    'absolute_all': int(10),
}

# npc coords relative to ego or junction, relative velocity or absolute velocity
npc_state_setting = ['to_ego', 'to_junction']


class StateManager3(StateManager):

    state_config = {
        'ego_state_len': int(4),  # default
        'npc_number': int(5),
        'npc_state_len': int(4),  # location and velocity in relative coordinate frame
    }

    def __init__(self,
                 carla_api,
                 ego_state_option='sumo',  # setting on ego state representation
                 # npc_state_option='to_ego',  # setting on NPC state representation  todo may be not necessary
                 # safe=False,
                 attention=False,
                 debug=False,
                 ):

        # reset ego state length
        self.ego_state_option = ego_state_option
        self.state_config['ego_state_len'] = ego_state_options[self.ego_state_option]

        # NPC vehicle state representation
        self.npc_state_option = 'to_junction' if ego_state_option == 'kinetics' else 'to_ego'

        super(StateManager3, self).__init__(carla_api,
                                            attention=attention,
                                            # safe=False,
                                            debug=debug,
                                            )

        # route of current episode, add a setter to get it
        self.ego_route = []
        self.start_waypoint = None
        self.end_waypoint = None  # not used for know
        self.route_length = None  # float

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

        # todo add args to select state representation option
        # get ego state according to ego state setting
        if self.ego_state_option == 'sumo':  # [position_code, speed]
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
            state = position_code + state
        elif self.ego_state_option == 'kinetics':  # [x, y, speed], coordinates relative to junction center
            junction_location = self.junction.bounding_box.location
            relative_loc = location - junction_location
            state = [relative_loc.x, relative_loc.y] + state
        elif self.ego_state_option == 'driven_dist':
            if self.ego_route:
                # distance ego vehicle has driven
                driven_distance, _ = get_distance_along_route(self.map, self.ego_route, location)
                # debug
                if driven_distance < self.route_length:
                    print('driven_distance: ', driven_distance)
                distance_proportion = driven_distance / self.route_length
            else:
                raise RuntimeError('Ego route not found!')
            state = [distance_proportion] + state

        if self.debug:
            print('ego location(2D): ', x, y)
            print('ego speed(km/h): ', speed)
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
        veh_velo = veh_info['velocity']
        rel_velo = veh_velo - ego_velo
        # use ndarray
        _rel_velo = np.array([rel_velo.x, rel_velo.y])
        # 2D velocity in ego coord frame
        relative_velocity = np.dot(self.T_ego2world, _rel_velo)  # in m/s

        # different npc state representation option
        if self.npc_state_option == 'to_ego':  # location and velocity relative to ego vehicle
            relative_location = veh_info['relative_location']
            npc_state = np.concatenate((relative_location, relative_velocity), axis=0)
            # in list
            npc_state = list(npc_state)
        elif self.npc_state_option == 'to_junction':
            junction_location = self.junction.bounding_box.location
            # abs coordinate in world system
            npc_location = veh_info['location']
            relative_location = npc_location - junction_location
            # velocity, in global coordinate system
            velocity = veh_info['velocity']
            npc_state = [relative_location.x, relative_location.y, velocity.x, velocity.y]
        else:
            raise RuntimeError('NPC state option is not set correctly.')

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
