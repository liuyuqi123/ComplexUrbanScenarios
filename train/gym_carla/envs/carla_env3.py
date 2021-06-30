"""
todo add argparser to get experiments settings: attention multi-task

This env is modified based on CarlaEnv2

Major improvements:

 - merge RL modules, with a more advanced state representation module for multi-task agent training

 - add args to set traffic flow module

 - add args to decide whether

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

import math
import time
import random
import numpy as np
from datetime import datetime
from collections import deque

import gym
from gym import spaces

from train.gym_carla.envs.BasicEnv import BasicEnv

from train.gym_carla.util_development.vehicle_control import Controller

# route module
# from gym_carla.scenario.route_design import generate_route_2
from train.gym_carla.modules.route_generator.junction_route_generator import RouteGenerator

# sensors for the ego vehicle
from train.gym_carla.util_development.sensors import Sensors

# ================   developing modules   ================
# ----------------   traffic flow module  ----------------
from train.gym_carla.modules.trafficflow.traffic_flow_manager import TrafficFlowManager

# ----------------  state representation module  ----------------
# from gym_carla.modules.rl_modules.state_representation.state_manager import StateManager
from train.gym_carla.modules.rl_modules.state_representation.state_manager_2 import StateManager

# ----------------  traffic light module  ----------------
from train.gym_carla.modules.traffic_lights.traffic_lights_manager import TrafficLightsController


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

# default junction center in Town03
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

# parameters for scenario
start_location = carla.Location(x=53.0, y=128.0, z=1.5)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)


class CarlaEnv3(gym.Env):

    # max target speed of ego vehicle
    ego_max_speed = 15  # in m/s

    # todo set by args
    simulator_timestep_length = 0.05
    max_episode_time = 60  # in seconds
    # max timestep limit of a episode
    max_episode_timestep = max_episode_time / simulator_timestep_length

    # available route_option must be contained in this tuple
    route_options = ('left', 'right', 'straight')

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
        'straight': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
    }

    def __init__(self,
                 attention=False,
                 multi_task=False,
                 route_option='left',
                 switch_traffic_flow=False,
                 # training=True,
                 carla_port=2000,
                 tm_port=int(8100),
                 debug=False,
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

        self.route_option = route_option
        # index number in route_options
        self.route_option_index = self.route_options.index(route_option)

        # todo add arg to set frequency
        # frequency of current simulation
        simulation_frequency = 1 / self.simulator_timestep_length
        self.carla_env = BasicEnv(town='Town03',
                                  host='localhost',
                                  port=carla_port,
                                  client_timeout=100.0,
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
        self.traffic_manager = self.carla_api['traffic_manager']

        # this api is responsible for traffic flow management
        # todo fix args api for init a traffic flow
        #  improve this part, current is for town03 experiment
        if self.multi_task:
            tf_direction = ['positive_x', 'negative_x', 'negative_y']
        else:
            if self.route_option == 'left':
                tf_direction = ['negative_y']
            elif self.route_option == 'right':
                tf_direction = ['negative_x', 'negative_y']
            else:  # straight
                tf_direction = ['positive_x', 'negative_y']

        # todo merge the api to set junction
        # set the traffic flow according to route option
        self.traffic_flow = TrafficFlowManager(self.carla_api, active_tf_direction=tf_direction)
        # set traffic flow sequence
        self.tf_sequence_index = 0
        self.traffic_flow_sequence = []
        self.set_tf_sequence()

        # route generation module
        self.route_manager = RouteGenerator(self.carla_api)

        # attributes for route sequence
        self.route_seq = []
        self.route_seq_index = None
        # set a route sequence for different task setting
        self.set_route_sequence()

        # todo use args or params to set action space
        # action and observation spaces
        self.discrete = False

        # state module
        self.state_manager = StateManager(self.carla_api, self.attention)
        self.observation_space = self.state_manager.observation_space  # a gym.Env attribute

        # self.simulator_timestep_length = 0.1
        self.episode_number = 0  # elapsed total episode number
        self.episode_step_number = 0  # elapsed timestep number of current episode
        self.elapsed_time = None  # simulation time of current episode

        # generate all available routes for the different route option
        # route format: <list>.<tuple>.(transform, RoadOption)
        for _route_option in self.route_options:
            ego_route, spawn_point, end_point = self.route_manager.get_route(junction_center,
                                                                             route_option=_route_option,
                                                                             )
            self.route_info[_route_option]['ego_route'] = ego_route
            self.route_info[_route_option]['spawn_point'] = spawn_point
            self.route_info[_route_option]['end_point'] = end_point

        # attributes about ego route
        self.ego_route = []
        self.spawn_point = None
        self.end_point = None

        self.controller = None
        self.controller_timestep_length = self.simulator_timestep_length  # todo use param to set
        self.use_pid_control = False  # if use PID controller for the ego vehicle control

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

        # traffic light manager
        self.tls_manager = TrafficLightsController(self.carla_api, junction=self.junction)
        # get traffic lights
        self.tls_manager.get_traffic_lights()

        # set the spectator on the top of junction
        self.carla_env.set_spectator_overhead(junction_center, yaw=270, h=70)

        # todo print other info, port number...
        print('A gym-carla env is initialized.')

    def init_spaces(self, params):
        """
        todo this method is not used yet, need to fix

        Init action and observation spaces.

        Consider behavioral action space, need to transform action to carla.VehicleControl
        """
        # action space
        if self.discrete:
            # a default action space dim is 4
            action_dim = 4
            action_space = spaces.Discrete(action_dim)
        else:
            action_high = np.array([1., 1.])
            action_low = np.array([0., 0.])
            action_space = spaces.Box(high=action_high, low=action_low, dtype=np.float32)

        # observation space
        # todo init with normalization clip and external module(attention multi-task)
        observation_space = None

        return action_space, observation_space

    def init_scenario(self):
        """
        todo future development.

        Generate scenario in according to config parameters.

        Init scenario in certain map and junction.
        :return:
        """
        pass

    def set_tf_sequence(self):
        """
        Set a sequence of active traffic flow.
        """
        # todo add setters to set tf seq
        #  the tf name must coordinate with definition in tf module
        # dict keys refers to ego route option
        traffic_flow_sequences = {
            'left': [
                ['negative_y'],
                ['positive_x'],
                ['negative_x'],
            ],
            'right': [
                ['negative_x'],
                ['negative_y'],
            ],
            'straight': [
                ['positive_x'],
                ['negative_x'],
                ['negative_y'],
            ]
        }

        if self.multi_task:
            # this seq is not used in multi-task training
            self.traffic_flow_sequence = [['positive_x', 'negative_x'], ['negative_y']]
        elif self.switch_traffic_flow:
            self.traffic_flow_sequence = traffic_flow_sequences[self.route_option]
        else:
            # first seq is most concerned situation
            _index = 0
            self.traffic_flow_sequence = traffic_flow_sequences[self.route_option][_index]

    def set_traffic_flow(self, clean_vehicles=False):
        """
        Set active traffic flow.
        traffic_flow_sequence stores active traffic flows.
        """
        # current active traffic flow
        active_traffic_flow = self.traffic_flow_sequence[self.tf_sequence_index]
        self.traffic_flow.set_active_tf_direction(active_traffic_flow, clean_vehicles=clean_vehicles)

        if self.tf_sequence_index == (len(self.traffic_flow_sequence)-1):
            self.tf_sequence_index = int(0)
        else:
            self.tf_sequence_index += 1

    def set_route_sequence(self):
        """
        Generate a sequence of route options.
         - for multi-task training, we will deploy a sequence consists 10 route.
         - for single-task training, we will run the specified route repeatedly
        """
        if self.multi_task:
            # todo add params to set seq length
            left_seq = ['left'] * 6
            right_seq = ['right'] * 3
            straight_seq = ['straight'] * 1
            self.route_seq = left_seq + right_seq + straight_seq
        else:  # single task condition
            self.route_seq = [self.route_option]

        self.route_seq_index = int(0)

    def set_ego_route(self):
        """
        Set ego route for agent when episode starts.

        This method must be called before spawning ego vehicle.
        """
        # todo fix multi-task training method
        if self.multi_task:
            self.route_option = self.route_seq[self.route_seq_index]
            if self.route_seq_index == (len(self.route_seq)-1):
                self.route_seq_index = 0
            else:
                self.route_seq_index += 1

        self.ego_route = self.route_info[self.route_option]['ego_route']
        self.spawn_point = self.route_info[self.route_option]['spawn_point']
        self.end_point = self.route_info[self.route_option]['end_point']

    def get_obs(self):
        """
        Get observation of current timestep through the state manager.
        """
        state = self.state_manager.get_state()
        # in list
        state = list(state)

        # append task code to the state vector
        if self.multi_task:
            # todo move this to a global attribute
            if self.route_option == 'left':
                task_code = [1, 0, 0, 1]
            elif self.route_option == 'right':
                task_code = [0, 0, 1, 1]
            else:  # straight
                task_code = [0, 1, 0, 1]
            # todo check if state is a list
            state += task_code

        # update state vector
        state = np.array(state)
        self.state = state

        return state

    def spawn_ego_vehicle(self):
        """
        Spawn ego vehicle.
        """
        # ego vehicle blueprint
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz2017'))
        # set color
        if bp.has_attribute('color'):
            color = '255, 0, 0'  # use string to identify a RGB color
            bp.set_attribute('color', color)

        # attributes of a blueprint
        # print('-'*10, 'blueprint attributes', '-'*10)
        # print(bp)
        # for attr in bp:
        #     print('  - {}'.format(attr))

        # set role name
        # hero is the very specified name to activate physic mode
        bp.set_attribute('role_name', 'hero')

        # attributes of a blueprint is stored as dict
        # ego_attri = self.ego_vehicle.attributes

        # set sticky control
        """
        carla Doc:
        when “sticky_control” is “False”, 
        the control will be reset every frame to 
        its default (throttle=0, steer=0, brake=0) values.
        """
        bp.set_attribute('sticky_control', 'True')

        if self.ego_route:
            start_trans = self.spawn_point.transform
            # spawn transform need to be checked, z value must larger than 0
            start_trans.location.z += 0.5
            try:
                self.ego_vehicle = self.world.spawn_actor(bp, start_trans)

                # time.sleep(0.1)
                self.try_tick_carla()

                # update ego vehicle id
                self.ego_id = self.ego_vehicle.id
                print('Ego vehicle is spawned.')
            except:
                raise Exception("Fail to spawn ego vehicle!")
        else:
            raise Exception("Ego route is not assigned!")

        # controller must be assigned to vehicle
        self.controller = Controller(self.ego_vehicle,
                                     self.controller_timestep_length)

        # add collision sensors
        self.ego_collision_sensor = Sensors(self.world, self.ego_vehicle)

        # time.sleep(0.1)
        self.try_tick_carla()

        # use try tick method
        # # todo use timestep length, wait for 1s
        # for i in range(20):
        #     self.try_tick_carla()
        # time.sleep(1.0)

        print('ego vehicle is ready.')

    def get_min_distance(self):
        """
        Get a minimum distance for waypoint buffer
        """
        if self.ego_vehicle:

            # fixme use current ego speed
            # speed = get_speed(self.ego_vehicle)
            #
            # target_speed = 4.0  # m/s
            # ref_speed = max(speed, target_speed)

            # in m/s
            ref_speed = 3

            # min distance threthold of waypoint reaching
            MIN_DISTANCE_PERCENTAGE = 0.75
            sampling_radius = ref_speed * 1.  # maximum distance vehicle move in 1 seconds
            min_distance = sampling_radius * MIN_DISTANCE_PERCENTAGE

            return min_distance
        else:
            raise

    def init_traffic_flow(self):
        """
        Check and wait till traffic flow is ready.
        """
        # todo set this by args or params?
        # minimum npc vehicle number to start
        min_npc_number = 5 if self.multi_task else 3

        # get TLS target phase for different route
        # phase = [0, 2] indicates WE/NS flow green light

        # todo add method to automatically get tls_cond through route_option
        if self.route_option in ['straight', 'left']:
            target_phase = [2, 3]   # the phase index is same as traffic light module
        else:  # right
            target_phase = [0, 1, 2]  # [0, 1, 2, 3]
            # todo in training phase, make it more difficult
            # if self.training:
            #     target_phase = [0, 1, 2, 3]
            # else:
            #     target_phase = [0, 1]

        vehNum_cond = False  # total vehicle number condition
        tls_cond = False  # traffic light state condition
        remain_time_cond = False  # remain time condition

        # todo add condition check for training and evaluation
        # if self.train_phase:
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]
        # else:
        #     # conditions = [vehNum_cond, tls_cond]  # huawei setting for evaluation
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]

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
            current_phase_index = self.tls_manager.get_phase_index()  # int
            # if current traffic light state phase is coordinate with target phase of the route option
            tls_cond = current_phase_index in target_phase

            if self.debug:
                if tls_cond:
                    print('')

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

        # reset active traffic flow if in single task training
        if self.switch_traffic_flow and not self.multi_task:
            self.set_traffic_flow()

        # prepare the traffic flow
        self.init_traffic_flow()
        # spawn ego only after traffic flow is ready
        self.spawn_ego_vehicle()

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

        return state

    def buffer_waypoint(self):
        """
        Buffering waypoints for planner.

        This method should be called each timestep.

        _waypoint_buffer: buffer some wapoints for local trajectory planning
        _waypoints_queue: all waypoints of current route

        :return: 2 nearest waypoint from route list
        """

        # add waypoints into buffer
        least_buffer_num = 10
        if len(self._waypoint_buffer) <= least_buffer_num:
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:  # when there is not enough waypoint in the waypoint queue
                    break

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, tuple in enumerate(self._waypoint_buffer):

            transform = tuple[0]
            # get transform and location
            self.ego_transform = self.ego_vehicle.get_transform()
            self.ego_location = self.ego_transform.location

            # current location of ego vehicle
            # self.ego_location = self.ego_vehicle.get_location()

            # todo check if z coord value effect the dist calculation
            _dist = self.ego_location.distance(transform.location)
            _min_dist = self.get_min_distance()

            # if no.i waypoint is in the radius
            if _dist < _min_dist:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):  # (max_index+1) waypoints to pop out of buffer
                self._waypoint_buffer.popleft()

    def map_action(self, action):
        """
        Map agent action to the vehicle control command.
        """
        # map action to vehicle target speed
        target_speed = self.ego_max_speed * (action[0] - 0.25 * action[1])

        # longitudinal speed
        target_speed = np.clip(target_speed, 0, self.ego_max_speed)

        return target_speed

    def step(self, action: tuple):
        """
        Take a timestep of simulation

        Use a PID controller to follow the target speed
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
                target_speed = 20  # m/s
            else:
                target_speed = self.map_action(action)

            # conduct control action by algorithm
            # todo add API for the PID controller
            # if self.use_pid_control == False:

            # map target speed to VehicleControl for a carla vehicle
            veh_control = self.controller.generate_control(target_speed=target_speed,
                                                           next_waypoint=self._waypoint_buffer[0])
        else:
            target_speed = 0.
            # hold break
            veh_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

            # todo test this, buffer last waypoint to avoid buffer empty
            # veh_control = self.controller.generate_control(target_speed=30,
            #                                                next_waypoint=self.end_point)  # carla.Waypoint

            print('route is finished, please terminate the simulation manually.')

        if self.debug:

            print('='*25)
            print('-'*8, 'timestep: ', self.episode_step_number, '-'*8)

            print('action: ', action)
            print('target speed: ', target_speed)

            print('Ego vehicle control: ')
            print('throttle: ', veh_control.throttle)
            print('steer: ', veh_control.steer)
            print('brake: ', veh_control.brake)

        self.ego_vehicle.apply_control(veh_control)

        # ================   Tick   ================
        # tick carla world and all modules
        self.tick_simulation()
        # ==========================================

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

    def try_tick_carla(self):
        """
        Tick the carla world with try exception method.

        In fixing fail to tick the world in carla
        """
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
                print('*-'*20)
                print('Fail to tick the world for once...')
                print('Last frame id is ', self.frame_id)
                print('*-' * 20)
                tick_counter += 1

        if tick_counter > 0:
            print('carla client is successfully ticked after ', tick_counter, 'times')

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

        # tick carla server
        self.try_tick_carla()

        # todo register all carla modules, add a standard api
        # ================   tick carla modules   ================

        # tick the traffic flow module
        self.traffic_flow.run_step()

        # traffic light module
        self.tls_manager.run_step()

        # update active npc vehicles
        self.update_vehicles()

        # todo print info in debug mode
        # print('-'*20)

    def update_vehicles(self):
        """
        Modified from BasicEnv.

        Only update NPC vehicles.
        """
        self.npc_vehicles = []

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a ActorList instance, iterable

        if vehicle_list:
            for veh in vehicle_list:
                attr = veh.attributes  # dict
                # filter ego vehicle by role name
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':
                    continue
                else:
                    self.npc_vehicles.append(veh)

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
            'time_exceed': -150.,
            'success': 100.,
            'step': -0.05,
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
            reward += 2*reward_dict['step']
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

        return reward, done, aux_info

    def destroy(self):
        """
        Destroy ego vehicle and its sensors

        todo test which carla API command works properly
        """
        delete_list = []  # put actors which need to be deleted into the list
        # todo sensors manipulation
        # if self.sensors:
        #     # self.sensors.destroy_sensors(self.client)
        #     self.sensors.destroy_sensors()

        # delete ego vehicle actor
        if self.ego_vehicle:
            # self.client.apply_batch([carla.command.DestroyActor(self.ego_car.id)])
            # if self.ego_vehicle.destroy():
            #     print('Ego vehicle is destroyed.')
            delete_list.append(self.ego_vehicle)

        # todo the api
        if self.ego_collision_sensor:
            for sensor in self.ego_collision_sensor.sensor_list:
                delete_list.append(sensor)

        # 所有要删除的actor保存在delete_list当中
        response_list = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in delete_list], True)

        if response_list:
            # 检查是否成功删除了全部车辆
            method_error = response_list[0].has_error()
            if method_error:
                print('actors fail to delete, please check.')
            else:
                print('actors are destroyed')

    # ================   future development   ================

    def render_local_map(self):
        """
        todo render a local map using pygame
        """
        pass

    def clear_world(self):
        """
        todo fix the API to clean the episode.

        Optional, clear the carla end NPCs.
        :return:
        """
        # actor_list = self.world.get_actors()
        # # clear traffic lights
        # if actor_list.filter('traffic.traffic_light'):
        #     # self.client.apply_batch([carla.command.DestroyActor(traffic_light) for traffic_light in actor_list.filter('traffic.traffic_light')])
        #     for traffic_light in actor_list.filter('traffic.traffic_light'):
        #         traffic_light.destroy()
        # # clear vehicles and walkers
        # if actor_list.filter('vehicle.*'):
        #     # self.client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in actor_list.filter('vehicle.*')])
        #     for vehicle in actor_list.filter('vehicle.*'):
        #         vehicle.destroy()
        # if actor_list.filter('walker.*'):
        #     # self.client.apply_batch([carla.command.DestroyActor(walker) for walker in actor_list.filter('walker.*')])
        #     for walker in actor_list.filter('walker.*'):
        #         walker.destroy()

        pass


def test():
    """
    Test usage of the class
    """

    env = CarlaEnv3(attention=True,
                    multi_task=False,
                    route_option='right',
                    switch_traffic_flow=True,
                    # training=True,
                    debug=True)

    obs = env.reset()

    step = 0
    while True:

        print('step: ', step + 1)

        # todo add action manager
        action = (1.0, 0.0)
        obs, reward, done, aux_info = env.step(action)

        # debug for npc state
        # env.test_actor_list()

        step += 1

        # time.sleep(0.1)
        print('')

        if done:
            obs = env.reset()


if __name__ == '__main__':

    test()
