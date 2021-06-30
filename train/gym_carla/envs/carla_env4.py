"""
This is a developing version based on CarlaEnv3.

Add new APIs to tune state representation.
"""

import math
import numpy as np
from datetime import datetime
from collections import deque
from line_profiler import line_profiler
import time

import carla

from train.gym_carla.envs.BasicEnv import BasicEnv

# route module
from train.gym_carla.modules.route_generator.junction_route_generator import RouteGenerator
from train.gym_carla.util_development.kinetics import get_transform_matrix

# sensors for the ego vehicle

# ================   developing modules   ================
# ----------------   traffic flow module  ----------------
from train.gym_carla.modules.trafficflow.traffic_flow_manager5 import TrafficFlowManager5

# ----------------  state representation module  ----------------
# from gym_carla.modules.rl_modules.state_representation.state_manager import StateManager
from train.gym_carla.modules.rl_modules.state_representation.state_manager_3 import StateManager3
from train.gym_carla.modules.rl_modules.state_representation.state_manager4 import StateManager4
from train.gym_carla.modules.rl_modules.state_representation.state_manager5 import StateManager5

# ----------------  traffic light module  ----------------
from train.gym_carla.modules.traffic_lights.traffic_lights_manager2 import TrafficLightsManager

from train.gym_carla.envs.carla_env3 import CarlaEnv3


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

# default junction center in Town03
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

# parameters for scenario
start_location = carla.Location(x=53.0, y=128.0, z=1.5)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)


class CarlaEnv4(CarlaEnv3):
    """
    <Down-sample>

    N / n = t / T = F/ f = down_sample_factor

    N, T, F refers to (simulation) system params
    n, t, f refers to RL module params

    """

    # down sample ratio
    down_sample_factor = 2
    down_sample_factor = math.ceil(down_sample_factor)

    # todo set by args
    simulator_timestep_length = 0.05
    max_episode_time = 60  # in seconds
    # maximum timestep limit of a episode
    rl_timestep = down_sample_factor * simulator_timestep_length
    max_episode_timestep = int(max_episode_time / rl_timestep)

    # max target speed of ego vehicle
    ego_max_speed = 15  # in m/s

    # available route_option must be contained in this tuple
    # 2 straight option, 0 -> left turn lane, 1 -> right turn lane
    route_options = ('left', 'right', 'straight_0', 'straight_1')

    # todo merge route option with route info
    # all available routes and spawn points
    route_info = {
        'left': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'right': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'straight_0': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'straight_1': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
    }

    # todo add an argparser to set port number
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
        self.traffic_flow = TrafficFlowManager5(self.carla_api, route_option)
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

    def set_ego_route(self):
        """
        Set ego route for agent in reset method, before each episode starts.

        This method must be called before spawning ego vehicle.
        """
        # todo fix multi-task training routes shifting
        if self.multi_task:
            self.route_option = self.route_seq[self.route_seq_index]
            if self.route_seq_index == (len(self.route_seq) - 1):
                self.route_seq_index = 0
            else:
                self.route_seq_index += 1

        self.ego_route = self.route_info[self.route_option]['ego_route']
        self.spawn_point = self.route_info[self.route_option]['spawn_point']
        self.end_point = self.route_info[self.route_option]['end_point']

        # ego route is required by some state representation options
        self.state_manager.set_ego_route(self.ego_route, self.spawn_point, self.end_point)

    def set_tf_sequence2(self):
        """
        For traffic flow manager4.

        :return:
        """
        # episode number each tf endures
        tf_period = int(5)
        tf_sequence = []

        # all available traffic flow
        available_tf_list = self.traffic_flow.active_tf_directions
        if len(available_tf_list) == 1:
            tf_sequence = [available_tf_list]
        elif len(available_tf_list) == 2:
            # todo set a minimum repetition for minor tf
            #  use a factor to tune the proportion of different tf selection
            _tf_sequence = [available_tf_list[0]] * tf_period + [available_tf_list[1]] * tf_period
            for tf in _tf_sequence:
                tf_sequence.append([tf])
        else:  # if more than 2 tf directions
            tf_sequence = [tf * tf_period for tf in available_tf_list]

        self.traffic_flow_sequence = tf_sequence

    def set_tf_sequence(self):
        """
        Original version, for traffic flow controlled by traffic manager.

        Set a sequence of active traffic flow.

        todo rewrite this method
        """
        # todo add setters to set tf seq
        #  the tf name must coordinate with definition in tf module
        # dict keys refers to ego route option
        traffic_flow_sequences = {
            'left': [
                ['negative_y_0', 'negative_y_1'],
                ['positive_x'],
                ['negative_x'],
            ],
            'right': [
                ['negative_x'],
                ['negative_y_0'],
            ],
            'straight_0': [
                # ['negative_y_0'],
                ['negative_x'],
            ],
            'straight_1': [
                # ['positive_x', 'negative_x'],
                ['positive_x'],
                ['negative_x'],
            ]
        }

        # todo re-design multi-task option, add hybrid traffic flow training option
        if self.multi_task:
            # this seq is not used in multi-task training
            self.traffic_flow_sequence = [['positive_x', 'negative_x'], ['negative_y_0', 'negative_y_1']]
        elif self.switch_traffic_flow:
            self.traffic_flow_sequence = traffic_flow_sequences[self.route_option]
        else:  # not switch and use single tf
            # first seq refers to a most concerned situation
            _index = 0
            self.traffic_flow_sequence = traffic_flow_sequences[self.route_option][_index]

            self.traffic_flow.set_active_tf_direction(self.traffic_flow_sequence, clean_vehicles=self.clean_tf)

            # # todo add args to set which traffic flow set we wish to use
            # if self.route_option == 'right':
            #     self.traffic_flow_sequence = [['negative_x', 'negative_y_0']]
            # else:  # todo straight agent may need manually set tf option
            #     # first seq refers to a most concerned situation
            #     _index = 0
            #     self.traffic_flow_sequence = traffic_flow_sequences[self.route_option][_index]

    def clear_traffic(self):
        """
        Clear all NPC vehicles and their collision sensor.
        """

        delete_actors = []
        # get actorlist
        vehicles = self.world.get_actors().filter('vehicle.*')
        for item in vehicles:
            delete_actors.append(item)

        # todo check if this line get actor correctly
        sensors = self.world.get_actors().filter('sensor.other.collision')
        for item in sensors:
            delete_actors.append(item)

        # delete actors
        actor_id = [x.id for x in delete_actors]
        # carla.Client.apply_batch_sync method
        response_list = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in delete_actors], True)
        failures = []
        for response in response_list:
            if response.has_error():
                failures.append(response)

        # if self.debug:
        #     if not failures:
        #         print('Following actors are destroyed.')
        #         print(actor_id)
        #     else:
        #         print('Fail to delete: ')
        #         print(failures)

    def set_traffic_flow2(self, clean_vehicles=False):
        """
        Set active traffic flow.

        This method is for traffic flow manager4.

        todo improve this method with clean previous vehicle
        """
        # current active traffic flow
        active_traffic_flow = self.traffic_flow_sequence[self.tf_sequence_index]
        # todo add a setter method in traffic flow class
        self.traffic_flow.active_tf_directions = active_traffic_flow

        if self.tf_sequence_index == (len(self.traffic_flow_sequence) - 1):
            self.tf_sequence_index = int(0)
        else:
            self.tf_sequence_index += 1

    def init_traffic_flow(self):
        """
        Reset traffic flow and check if current traffic flow is ready.
        """

        # ========================================================
        # ---------------   switch traffic flow   ----------------

        # --- original version ---

        # reset active traffic flow if in single task training
        # if self.switch_traffic_flow:
        #     if (self.episode_number + 1) % self.tf_switch_period == 0:
        #         self.set_traffic_flow(clean_vehicles=self.clean_tf)
        # ========================================================

        # --- traffic flow manager4 ---
        # switch traffic flow
        self.set_traffic_flow2()
        print('')

        # ========================================================

        # In some experiments, the spawn point is occupied
        # by a holding vehicle and we don't know why.
        # To avoid this, reset active traffic flow and
        # clear traffic periodically.
        # todo add args to set period length
        clear_period = 100
        if (self.episode_number + 1) % clear_period == 0:
            # todo test this
            self.traffic_flow.clean_up()
            # in case any NPC vehicles are not destroyed
            self.clear_traffic()

        # todo set this by args or params?
        # minimum npc vehicle number to start
        min_npc_number = 5 if self.multi_task else 3

        # get target traffic lights phase according to route_option
        # phase = [0, 2] refers to x and y direction Green phase respectively
        if self.route_option in ['straight', 'left']:
            target_phase = [2, 3]  # the phase index is same as traffic light module
        else:  # right
            target_phase = [0, 1, 2]  # [0, 1, 2, 3]
            # todo in training phase, make it more difficult
            # if self.training:
            #     target_phase = [0, 1, 2, 3]
            # else:
            #     target_phase = [0, 1]

        # todo add condition check for training and evaluation
        # if self.train_phase:
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]
        # else:
        #     # conditions = [vehNum_cond, tls_cond]  # huawei setting for evaluation
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]

        # ==============   init conditions   ==============
        # conditions for init traffic flow finishes
        # vehicle number condition
        self.update_vehicles()
        vehicle_numbers = len(self.npc_vehicles)
        vehNum_cond = vehicle_numbers >= min_npc_number

        # traffic light state condition
        if self.use_tls_control:
            current_phase_index = self.tls_manager.get_phase_index()  # index, int
            # if current tls phase coordinates with target phase
            tls_cond = current_phase_index in target_phase
        else:
            tls_cond = True

        # remain time condition, additional to tls cond
        # remain_time_cond = False

        # conditions in a list
        conditions = [vehNum_cond, tls_cond]

        while not all(conditions):
            # tick the simulator before check conditions
            self.tick_simulation()

            # ==============  vehicle number condition  ==============
            self.update_vehicles()
            vehicle_numbers = len(self.npc_vehicles)
            vehNum_cond = vehicle_numbers >= min_npc_number

            if self.debug:
                if vehicle_numbers >= min_npc_number:
                    print('npc vehicle number: ', vehicle_numbers)

            # ==============  traffic light state condition  ==============
            # get tls phase from tls module
            if self.use_tls_control:
                current_phase_index = self.tls_manager.get_phase_index()  # index, int
                # if current tls phase coordinates with target phase
                tls_cond = current_phase_index in target_phase

            # if self.debug:
            #     if tls_cond:
            #         print('')

            # todo add condition on the remain time of current tls phase
            # # remain time is considered in training phase
            # if tls_cond and vehNum_cond:
            #     remain_time = traci.trafficlight.getNextSwitch('99810003') - traci.simulation.getTime()
            #     remain_time_cond = remain_time >= 36.5
            #
            #     remain_time_cond = remain_time >= 36.9  # original
            #     remain_time_cond = remain_time <= 0.1  # debug waiting before junction
            #
            #     if using interval, interval module is required
            #     zoom = Interval(25., 37.)
            #     remain_time_cond = remain_time in zoom

            # ==============  append all conditions  ==============
            conditions = [vehNum_cond, tls_cond]

    def set_velocity(self, vehicle, target_speed: float):
        """
        Set a vehicle to the target velocity.

        params: target_speed: in m/s
        """

        transform = vehicle.get_transform()
        # transform matrix Actor2World
        trans_matrix = get_transform_matrix(transform)

        # target velocity in world coordinate system
        target_vel = np.array([[target_speed], [0.], [0.]])
        target_vel_world = np.dot(trans_matrix, target_vel)
        target_vel_world = np.squeeze(target_vel_world)

        # carla.Vector3D
        target_velocity = carla.Vector3D(
            x=target_vel_world[0],
            y=target_vel_world[1],
            z=target_vel_world[2],
        )
        #
        vehicle.set_target_velocity(target_velocity)

        # tick twice to reach target speed
        # for i in range(2):
        #     self.try_tick_carla()

    def get_carla_time(self):
        """
        Get carla simulation time.
        :return:
        """

        # reset carla
        snapshot = self.world.get_snapshot()
        frame = snapshot.timestamp.frame
        elapsed_seconds = snapshot.timestamp.elapsed_seconds

        return frame, elapsed_seconds

    def map_action(self, action):
        """
        Map agent action to the vehicle control command.

        todo map action to [-1, 1], by setting decel_ratio = 1.
        """
        decel_ratio = 0.25
        # map action to vehicle target speed
        target_speed = action[0] - decel_ratio * action[1]
        target_speed = np.clip(target_speed, 0., 1.)
        target_speed = 3.6 * self.ego_max_speed * target_speed  # in km/h

        return target_speed

    def compute_reward(self):
        """
        Compute reward value of current timestep

        :return:
        """
        # todo fix method for multi-task
        reward = 0.

        # todo add api to set reward for tuning
        reward_dict = {
            'collision': -350.,
            'time_exceed': -100.,
            'success': 150.,
            'step': -0.3,
        }

        # todo fix aux info for stable baselines
        # collision = False
        # time_exceed = False
        aux_info = 'running'

        done = False

        # check if collision happens by collision sensor of ego vehicle
        collision = self.collision_flag
        # check time exceed
        time_exceed = self.episode_step_number > self.max_episode_timestep

        # compute reward value based on above
        if collision:
            print('Failure: collision!')
            reward += reward_dict['collision']
            done = True
            aux_info = 'collision'
        elif time_exceed:
            print('Failure: Time exceed!')
            reward += reward_dict['time_exceed']
            done = True
            aux_info = 'time_exceed'
        else:  # check if reach goal
            # use distance between ego vehicle and end_point
            # to check if reach the goal
            dist_threshold = 5.0
            ego_loc = self.ego_vehicle.get_location()
            end_loc = self.end_point.transform.location
            dist_ego2end = ego_loc.distance(end_loc)

            if dist_ego2end <= dist_threshold:
                done = True
                aux_info = 'success'
                reward += reward_dict['success']
                print('Success: Ego vehicle reached goal.')

        # step cost
        if self.episode_step_number > 0.5 * self.max_episode_timestep:
            reward += 2 * reward_dict['step']
        else:
            reward += reward_dict['step']

        # accumulative reward
        self.episode_reward += reward

        if done:
            # measure episode time by counting step numbers
            self.episode_time = self.episode_step_number * self.simulator_timestep_length

            # todo add debug option to print episode time
            #  check if elapsed time equals to episode time

            print('episode reward: ', self.episode_reward)
            self.episode_reward = 0

            # time
            end_frame, end_elapsed_seconds = self.get_carla_time()
            self.end_frame, self.end_elapsed_seconds = end_frame, end_elapsed_seconds
            episode_frame = self.end_frame - self.start_frame
            episode_elapsed_seconds = self.end_elapsed_seconds - self.start_elapsed_seconds

        return reward, done, aux_info

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

        # todo add args and APIs to tune params' range of traffic flow
        # ==========   set collision probability   ==========
        # only TrafficFlowManager5 has this API
        if self.traffic_flow.__class__.__name__ == 'TrafficFlowManager5':
            collision_prob_range = [0.75, 0.95]
            # todo add args to set decay range
            collision_decay_length = int(1000)  # episode number for collision prob increasing
            # default episode number is 2000
            collision_prob = collision_prob_range[0] + \
                self.episode_number * (collision_prob_range[1] - collision_prob_range[0]) / collision_decay_length
            collision_prob = np.clip(collision_prob, collision_prob_range[0], collision_prob_range[1])
            self.traffic_flow.set_collision_probability(collision_prob)

        return state

    def try_tick_carla(self):
        """
        Tick the carla world with try exception method.

        In fixing fail to tick the world in carla
        """
        # test tick
        # self.world.tick(100.)

        max_try = int(100)
        tick_success = False
        tick_counter = int(0)
        while not tick_success:
            if tick_counter >= max_try:
                raise RuntimeError('Fail to tick carla for ', max_try, ' times...')
            try:
                # if this step success, a frame id will return
                frame_id = self.world.tick()
                if frame_id:
                    self.frame_id = frame_id
                    tick_success = True
            except:  # for whatever the error is..
                print('*-' * 20)
                print('Fail to tick the world for once...')
                print('Last frame id is ', self.frame_id)
                print('*-' * 20)
                tick_counter += 1

        if tick_counter > 0:
            print('carla client is successfully ticked after ', tick_counter, 'times')

    def get_frame(self):
        """
        Get frame id from world.

        :return: frame_id
        """
        snapshot = self.world.get_snapshot()
        frame = snapshot.timestamp.frame
        elapsed_seconds = snapshot.timestamp.elapsed_seconds

        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        return frame, elapsed_seconds

    def tick_simulation(self):
        """
        Tick the simulation.

        The following modules will be ticked.

        todo manage modules with a manager, and tick all registered modules using loop
        :return:
        """

        # todo fix if controller and simulation has different timestep
        # ===================  co-simulation code  ===================
        # for _ in range(int(self.control_dt/self.server_dt)):
        #     # world tick
        #     start = time.time()
        #     self.world.tick()
        #     end = time.time()
        #     elapsed = end - start
        #     if elapsed < self.server_dt:
        #         time.sleep(self.server_dt - elapsed)

        # todo render in local map
        # ===================   render local map in pygame   ===================
        #
        # vehicle_poly_dict = self.localmap.get_actor_polygons(filter='vehicle.*')
        # self.all_polygons.append(vehicle_poly_dict)
        # while len(self.all_polygons) > 2: # because two types(vehicle & walker) of polygons are needed
        #     self.all_polygons.pop(0)
        # # pygame render
        # self.localmap.display_localmap(self.all_polygons)

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # tick carla server
        self.try_tick_carla()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # todo register all carla modules, add a standard api
        # ================   tick carla modules   ================

        # tick the traffic flow module
        self.traffic_flow.run_step()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # traffic light module
        self.tls_manager.run_step()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # update active npc vehicles
        self.update_vehicles()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

    def reset_episode_count(self):
        """
        Reset class attribute episode_number.
        """
        self.episode_number = 0

    def step(self, action: tuple):
        """
        Run a step for RL training.

        This method will:

         - tick simulation multiple times according to down_sample_factor
         - tick carla world and all carla modules
         - apply control on ego vehicle
         - compute rewards and check if episode ends
         - update information

        :param action:
        :return:
        """

        # ==================================================
        # if using a pid controller
        # ==================================================
        self.buffer_waypoint()

        # todo check sticky control usage in sync mode
        if self._waypoint_buffer:
            # constant speed for debug
            constant_speed = False
            if constant_speed:
                target_speed = self.ego_max_speed * 3.6  # m/s
            else:  # conduct control action by algorithm
                target_speed = self.map_action(action)

            # map target speed to VehicleControl for a carla vehicle
            veh_control = self.controller.generate_control(target_speed=target_speed,
                                                           next_waypoint=self._waypoint_buffer[0])
        else:
            target_speed = 0.
            # hold brake
            veh_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

            # todo test this, buffer last waypoint to avoid buffer empty
            # veh_control = self.controller.generate_control(target_speed=30,
            #                                                next_waypoint=self.end_point)  # carla.Waypoint

            print('route is finished, please terminate the simulation manually.')

        if self.debug:
            # print('='*30)
            print('-' * 8, 'timestep ', self.episode_step_number, '-' * 8)

            print('action: ', action)
            print('target speed: ', target_speed)

            print('Ego vehicle control: ')
            print('throttle: ', veh_control.throttle)
            print('steer: ', veh_control.steer)
            print('brake: ', veh_control.brake)

        self.ego_vehicle.apply_control(veh_control)

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # ================   Tick simulation   ================
        # todo ref carla co-sim method with time.time() to sync with real world time
        # tick carla world and all modules
        for i in range(self.down_sample_factor):
            self.tick_simulation()
        # =====================================================

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # todo add getter method
        # check collision flag
        self.collision_flag = self.ego_collision_sensor.collision_flag

        # test state manager module
        state = self.get_obs()

        # todo get kinetic information from state manager
        # if self.debug:
        #     ego_acceleration = self.ego_vehicle.get_acceleration()
        #     ego_velocity = self.ego_vehicle.get_velocity()
        #
        #     accel_norm = math.sqrt(ego_acceleration.x ** 2 + ego_acceleration.y ** 2 + ego_acceleration.z ** 2)
        #     speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
        #
        #     # print('ego acceleration: [', ego_acceleration.x, ego_acceleration.y, ego_acceleration.z, ']')
        #     print('acceleration norm: ', accel_norm)
        #     # print('ego velocity: [', ego_velocity.x, ego_velocity.y, ego_velocity.z, ']')
        #     print('speed: ', speed)

        # get reward of current step
        reward, done, info = self.compute_reward()
        aux = {'exp_state': info}

        # state, reward, done, aux_info = None, None, None, None

        # update the step number
        if not done:
            self.episode_step_number += 1

        return state, reward, done, aux
