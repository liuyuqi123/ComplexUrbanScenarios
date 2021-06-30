"""
todo merge all modifications

A fixed version of CarlaEnv4.

Major modifications:

 - To test modifications on traffic flow module, add args and methods for traffic flow params decay

 -

"""

import math
import numpy as np
from datetime import datetime
from collections import deque
from line_profiler import line_profiler
import time

import carla

from gym_carla.envs.BasicEnv import BasicEnv

# route module
# from gym_carla.scenario.route_design import generate_route_2
from gym_carla.modules.route_generator.junction_route_generator import RouteGenerator

from gym_carla.util_development.kinetics import get_transform_matrix

# sensors for the ego vehicle

# ================   developing modules   ================
# ----------------   traffic flow module  ----------------
# from gym_carla.envs.trafficflow import TrafficFlow
# from gym_carla.modules.trafficflow.trafficflow_3 import TrafficFlow
# from gym_carla.modules.trafficflow.traffic_flow_tunable import TrafficFlowTunable
# from gym_carla.modules.trafficflow.traffic_flow_tunable2 import TrafficFlowTunable2
from gym_carla.modules.trafficflow.traffic_flow_manager2 import TrafficFlowManager2
from gym_carla.modules.trafficflow.traffic_flow_manager4 import TrafficFlowManager4
from gym_carla.modules.trafficflow.traffic_flow_manager5 import TrafficFlowManager5

# ----------------  state representation module  ----------------
# from gym_carla.modules.rl_modules.state_representation.state_manager import StateManager
from gym_carla.modules.rl_modules.state_representation.state_manager_3 import StateManager3
from gym_carla.modules.rl_modules.state_representation.state_manager4 import StateManager4
from gym_carla.modules.rl_modules.state_representation.state_manager5 import StateManager5

# ----------------  traffic light module  ----------------
from gym_carla.modules.traffic_lights.traffic_lights_manager2 import TrafficLightsManager

from gym_carla.envs.carla_env3 import CarlaEnv3
from gym_carla.envs.carla_env4 import CarlaEnv4


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

# default junction center in Town03
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

# parameters for scenario
start_location = carla.Location(x=53.0, y=128.0, z=1.5)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)


class CarlaEnv4Fixed(CarlaEnv4):
    """"""

    def __init__(self,
                 carla_port=2000,
                 tm_port=int(8100),
                 route_option='left',
                 state_option='sumo',
                 tm_seed=int(0),
                 initial_speed=None,
                 use_tls_control=True,
                 switch_traffic_flow=False,
                 multi_task=False,
                 attention=False,
                 debug=False,
                 # training=True,
                 # ==========   new args for traffic flow   ==========
                 tf_randomize=True,  # True for training, False to run with fixed traffic flow
                 collision_prob_decay=True,  # if True, enable collision prob decay
                 tf_params_decay=True,  # traffic params decay, shrink to a compact range with training proceeding
                 ):
        """
        Init the RL env for agent training.

        :param attention:
        :param multi_task:
        :param route_option:
        """

        # todo add args to run training or evaluating
        # self.train_phase = True

        # in single task training, whether switch traffic flow of different directions
        self.switch_traffic_flow = switch_traffic_flow

        # debug mode to visualization and printing
        self.debug = debug

        self.attention = attention
        self.multi_task = multi_task

        # set a initial speed to ego vehicle
        self.initial_speed = initial_speed

        self.route_option = route_option
        # index number in route_options
        self.route_option_index = self.route_options.index(route_option)

        # todo add arg to set frequency
        # frequency of current simulation
        simulation_frequency = 1 / self.simulator_timestep_length

        # todo use args to set map for different scenarios
        self.carla_env = BasicEnv(town='Town03',
                                  host='localhost',
                                  port=carla_port,
                                  client_timeout=100.0,  # todo add into args
                                  frequency=simulation_frequency,
                                  tm_port=tm_port,  # need to set to 8100 on 2080ti server
                                  )

        # sync mode: requires manually tick simulation
        self.carla_env.set_world(sync_mode=True)

        # get API
        self.carla_api = self.carla_env.get_env_api()
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']

        # todo test usage of seed
        self.traffic_manager = self.carla_api['traffic_manager']

        # this api is responsible for traffic flow management
        # todo fix args api for init a traffic flow
        #  improve this part, current is for town03 experiment
        if self.multi_task:
            tf_direction = ['positive_x', 'negative_x', 'negative_y_0', 'negative_y_1']
        else:
            if self.route_option == 'left':
                tf_direction = ['negative_y_0', 'negative_y_1']
            elif self.route_option == 'right':
                tf_direction = ['negative_x', 'negative_y_0']
            elif self.route_option == 'straight_0':  # go straight on left turn lane
                tf_direction = ['negative_y_0']
            elif self.route_option == 'straight_1':  # go straight on right turn lane
                tf_direction = ['positive_x']
            else:
                raise ValueError('The given route option is not found in route options, please check.')

        # todo fix api set junction for carla modules
        # todo set the traffic flow according to route option

        # ================   In developing traffic flow module   ================
        # # use traffic manager to control traffic flow
        # self.traffic_flow = TrafficFlowManager2(self.carla_api,
        #                                         active_tf_direction=tf_direction,
        #                                         tm_seed=tm_seed
        #                                         )
        # # set traffic flow sequence
        # self.tf_sequence_index = 0
        # # todo fix attribute name
        # # active traffic flow in list
        # self.traffic_flow_sequence = []
        # self.set_tf_sequence()  # this method cooperate with
        # -----------------------------------------------------------------------
        # ----------------   Traffic flow manager4   ----------------
        # traffic flow controlled by planner and controller
        # self.traffic_flow = TrafficFlowManager4(self.carla_api, route_option)

        self.tf_randomize = tf_randomize
        self.traffic_flow = TrafficFlowManager5(
            carla_api=self.carla_api,
            route_option=route_option,
            randomize=self.tf_randomize,  # set by init arg
        )

        self.collision_prob_decay = collision_prob_decay
        if not self.collision_prob_decay:
            collision_prob = 1.  # todo add arg to set this value
            self.traffic_flow.set_collision_probability(collision_prob)

        self.tf_params_decay = tf_params_decay

        self.tf_sequence_index = 0

        self.traffic_flow_sequence = []
        self.set_tf_sequence2()

        # todo use args or param dict to set traffic flow and traffic lights logic
        # if using switch_traffic_flow mode, active traffic flow will be reset after the period episodes
        self.tf_switch_period = int(10)
        self.clean_tf = False  # if clean current traffic flow when tf is reset

        # =======================================================================

        # route generation module
        self.route_manager = RouteGenerator(self.carla_api)
        # todo add args to set route distance
        self.route_manager.set_route_distance((15, 5.))  # set to 3. for right turn task

        # attributes for route sequence
        self.route_seq = []
        self.route_seq_index = None
        # set a route sequence for different task setting
        self.set_route_sequence()

        # todo use args or params to set action space
        # action and observation spaces
        self.discrete = False

        # state manager
        # fixme fix this after finish tuning
        # check state option exists
        # if ego_state_option not in ego_state_options.keys():
        #     raise ValueError('Wrong state option, please check')

        # select state manager through state option
        if state_option == 'sumo':  # the successful left task
            self.state_manager = StateManager3(self.carla_api,
                                               ego_state_option=state_option,  # fixme fix statemanager3 args
                                               attention=self.attention,
                                               debug=self.debug,  # self.debug
                                               )
        elif state_option == 'absolute_all':  # new state manager consists more info
            self.state_manager = StateManager4(self.carla_api,
                                               state_option=state_option,
                                               attention=self.attention,
                                               debug=self.debug,
                                               )
        elif state_option == 'sumo_1':
            self.state_manager = StateManager5(self.carla_api,
                                               attention=self.attention,
                                               debug=self.debug,
                                               )
        else:
            raise ValueError('State option is wrong, please check.')

        # todo fix ob space
        # self.observation_space = self.state_manager.observation_space  # a gym.Env attribute

        self.episode_number = 0  # elapsed total episode number
        self.episode_step_number = 0  # elapsed timestep number of current episode
        self.elapsed_time = None  # simulation time of current episode

        # generate all available routes for the different route option
        # route format: <list>.<tuple>.(transform, RoadOption)
        for _route_option in self.route_options:
            ego_route, spawn_point, end_point = self.route_manager.get_route(junction_center,
                                                                             route_option=_route_option)
            self.route_info[_route_option]['ego_route'] = ego_route
            self.route_info[_route_option]['spawn_point'] = spawn_point
            self.route_info[_route_option]['end_point'] = end_point

        # attributes about ego route
        self.ego_route = []
        self.spawn_point = None
        self.end_point = None

        self.controller = None
        # todo need to check
        self.controller_timestep_length = self.simulator_timestep_length * self.down_sample_factor
        self.use_pid_control = False  # if use PID controller for the ego vehicle control

        # todo merge waypoints buffer into local planner
        # ==================================================
        # ----------       waypoint buffer       ----------
        # ==================================================
        # waypoint buffer for navigation and state representation
        # a queue to store original route
        self._waypoints_queue = deque(maxlen=100000)  # maximum waypoints to store in current route
        # buffer waypoints from queue, get nearby waypoint
        self._buffer_size = 50
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # todo near waypoints is waypoint popped out from buffer
        # self.near_waypoint_queue = deque(maxlen=50)
        # ==================================================

        # ego vehicle
        self.ego_vehicle = None  # actor
        self.ego_id = None  # id is set by the carla, cannot be changed by user
        self.ego_location = None
        self.ego_transform = None

        self.ego_collision_sensor = None  # collision sensor for ego vehicle
        # todo add a manual collision check
        self.collision_flag = False  # flag to identify a collision with ego vehicle

        #
        self.npc_vehicles = []  # active NPC vehicles
        self.actor_list = []  # list contains all carla actors

        # state of current timestep
        self.state = None

        # this is a debugger for carla world tick
        self.frame_id = None

        # accumulative reward of this episode
        self.episode_reward = 0
        self.episode_time = None

        # todo add these methods to parent carla modules
        # get junction from route manager
        self.junction = self.route_manager.get_junction_by_location(junction_center)

        # set junction for the state manager
        self.state_manager.set_junction(self.junction)

        self.use_tls_control = use_tls_control
        # traffic light manager
        self.tls_manager = TrafficLightsManager(self.carla_api,
                                                junction=self.junction,
                                                use_tls_control=self.use_tls_control,
                                                )

        # set the spectator on the top of junction
        self.carla_env.set_spectator_overhead(junction_center, yaw=270, h=70)

        # debug tf
        # debug_location = carla.Location(x=-6.411462, y=68.223877, z=0.200000)
        # self.carla_env.set_spectator_overhead(debug_location, yaw=270, h=50)

        # time debuggers
        self.start_frame, self.end_frame = None, None
        self.start_elapsed_seconds, self.end_elapsed_seconds = None, None

        # todo print other info, port number...
        print('A gym-carla env is initialized.')

    def reset(self):
        """
        Reset ego vehicle.
        :return:
        """
        # destroy ego vehicle and its sensors
        if self.ego_vehicle:  # check if exist
            self.destroy()

        # clear buffered waypoints
        self._waypoints_queue.clear()
        self._waypoint_buffer.clear()

        # reset ego route
        self.set_ego_route()

        # prepare the traffic flow
        self.init_traffic_flow()
        # spawn ego only after traffic flow is ready
        self.spawn_ego_vehicle()

        # set initial speed
        if self.initial_speed:
            self.set_velocity(self.ego_vehicle, self.initial_speed)

        # todo move to spawn vehicle
        # buffer waypoints from ego route
        for elem in self.ego_route:
            self._waypoints_queue.append(elem)

        # get state vector from manager
        state = self.get_obs()

        print('=' * 25)
        print('episode_number: ', self.episode_number)
        print('episode_step_number:', self.episode_step_number)
        print('=' * 25)

        # reset step number at the end of the reset method
        self.episode_step_number = 0
        self.episode_number += 1  # update episode number

        # update start time
        start_frame, start_elapsed_seconds = self.get_carla_time()
        self.start_frame, self.start_elapsed_seconds = start_frame, start_elapsed_seconds

        # todo move this part into a new method
        # todo add args and APIs to tune params' range of traffic flow
        # ==========   set collision probability   ==========
        """
        TrafficFlowManager5 and TrafficFlowManager5Fixed has such API.
        
        TrafficFlowManager5Fixed is a developing version.
        
        Instruction:
        There are several mode of traffic flow settings:
        
         - fixed traffic flow param
         - whether use stochastic process to generate traffic params
        
        """
        if self.traffic_flow.__class__.__name__ in ['TrafficFlowManager5']:
            if self.collision_prob_decay:
                collision_prob_range = [0.75, 0.95]
                # todo add args to set decay range
                collision_decay_length = int(1000)  # episode number for collision prob increasing
                # default episode number is 2000
                collision_prob = collision_prob_range[0] + \
                    self.episode_number * (collision_prob_range[1] - collision_prob_range[0]) / collision_decay_length
                collision_prob = np.clip(collision_prob, collision_prob_range[0], collision_prob_range[1])
                self.traffic_flow.set_collision_probability(collision_prob)

            # traffic flow params decay
            if self.tf_params_decay:
                # this setting is available when param noise is enabled
                if self.tf_randomize:
                    # decay length, episode number, 5000 is current minimum training length
                    # tf_param_decay_length = int(5000)

                    # for debug
                    tf_param_decay_length = int(10)

                    # todo store params range through external data storage
                    #  or retrieve traffic flow params form the class
                    # for now the target params are set manually here
                    final_speed_range = (25, 40)
                    final_distance_range = (10, 25)

                    # retrieve original traffic flow params
                    for tf in self.traffic_flow.active_tf_directions:
                        info_dict = self.traffic_flow.traffic_flow_info[tf]
                        # original range
                        speed_range = info_dict['target_speed_range']
                        distance_range = info_dict['distance_range']

                        # new range
                        current_speed_range = (
                            self.linear_mapping(speed_range[0], final_speed_range[0], tf_param_decay_length, self.episode_number),
                            self.linear_mapping(speed_range[1], final_speed_range[1], tf_param_decay_length, self.episode_number),
                        )
                        current_distance_range = (
                            self.linear_mapping(distance_range[0], final_distance_range[0], tf_param_decay_length, self.episode_number),
                            self.linear_mapping(distance_range[1], final_distance_range[1], tf_param_decay_length, self.episode_number),
                        )

                        self.traffic_flow.traffic_flow_info[tf]['target_speed_range'] = current_speed_range
                        self.traffic_flow.traffic_flow_info[tf]['distance_range'] = current_distance_range

        return state

    @staticmethod
    def linear_mapping(original_value: float, target_value: float, total_step: int, current_step: int):
        """
        One-dimensional linear mapping.

        Shrinking of traffic flow params

        :return: current_range
        """
        delta_y = target_value - original_value
        k = delta_y / total_step
        b = original_value

        current_value = k * current_step + b

        return current_value
