"""
An RL based autonomous agent for scenario verification.

The agent is tested using scenario.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque

import carla

# CAUTION! Must append scenario runner repo path in run_scenario_test
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.timer import GameTime
from srunner.tools.route_manipulation import downsample_route

from train.gym_carla.util_development.vehicle_control import Controller
from train.gym_carla.util_development.kinetics import get_transform_matrix, get_inverse_transform_matrix, vector2array
from train.gym_carla.util_development.util_visualization import draw_waypoint
# from train.gym_carla.util_development.carla_color import *

# ================   Import Actor of the RL   ================
from train.rl_agents.td3.single_task.without_attention.rl_utils import ActorNetwork

# ==========   state representation module   ==========
# from gym_carla.modules.rl_modules.state_representation.state_manager4 import StateManager4
from train.gym_carla.modules.rl_modules.state_representation.state_manager5 import StateManager5
from train.gym_carla.envs.carla_env4 import CarlaEnv4


# config parameters for state module
rl_config = {
    # not used for now
    'ego_feature_num': int(4),  # with a code respond to current location, for attention query
    'npc_number': int(5),
    'npc_feature_num': int(6),

    'state_size': int(34),  # sumo_1, 4 + 5*6
    'action_size': int(2),
    'tau': 1e-3,
    'lra': 2e-5,
}


def init_tensorflow():
    """
    init tensorflow.
    """
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


class TestAgent(AutonomousAgent):

    def __init__(self, path_to_conf_file):
        """
        Load trained agent.

        todo add method to select model based on route option
        """
        # original method
        super(TestAgent, self).__init__(path_to_conf_file)

        self.debug = False

        self.actor = None
        self.saver = None
        self.sess = None

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

        # todo fix this, set client for agent when agent_instance initialization
        # set carla.Client for the agent
        # self.client = CarlaDataProvider.get_client()
        self.client = None
        self.world = None
        self.map = None
        self.carla_api = None
        self.state_manager = None

        # original
        self._global_plan_world_coord = []
        self._global_plan = []

        self.junction = None

        # list to store route of waypoints
        self.route = []

        self.ego_vehicle = None
        self.controller = None

        self.state = None
        self.action = None

    def load_model(self, route_option, manual=False, debug=False):
        """
        Restore RL model with option.
        """
        self.debug = debug
        if debug:
            print(
                '-'*50, '\n',
                'Running Agent in debug mode. Random policy will be used.', '\n',
                '-' * 50,
            )
            return

        # manually select model
        if manual:
            model_path = ''

            print('Please make sure route option is coordinate.')
        else:
            model_base = os.path.join(os.environ['GYM_CARLA_ROOT'],
                                      'rl_agents/challenge_agents/test_agents/model_dict',
                                      route_option)

            model_name = None
            files = os.listdir(model_base)
            for file in files:
                if file.split('.')[-1] == 'meta':
                    model_name = file.split('.')[0] + '.ckpt'
                    break
                else:
                    continue
            if not model_name:
                raise ValueError('model dict not found, please check.')

            model_path = os.path.join(model_base, model_name)

        # restore NN through model path
        self.restore_network(model_path)
        print(
            '-' * 50, '\n',
            'Model', model_path, 'is successfully restored.', '\n',
            '-' * 50,
        )

    def restore_network(self, model_path):
        """
        Restore rl network or using debug mode.
        """
        self.actor = ActorNetwork(rl_config['state_size'],
                                  rl_config['action_size'],
                                  rl_config['tau'],  # not used in forward propagation
                                  rl_config['lra'],  # not used in forward propagation
                                  'actor')  # str, name of the NN

        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(self.sess, model_path)
        # saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')

    def set_client(self, client: carla.Client, tm_port=int(8100)):
        """
        Set carla.Client for the agent instance.

        Init necessary modules.
        """
        self.client = client
        # get carla API for initialization
        self.carla_api = self.get_carla_api(self.client, tm_port=tm_port)

        self.state_manager = StateManager5(self.carla_api)

    def set_junction(self, junction: carla.Junction):
        """
        Set junction instance for the state manager.

        This method must be called after the state manager is set.
        """
        if self.state_manager:
            self.state_manager.set_junction(junction)

            # add this to the class attribute in case having other usages
            self.junction = junction
        else:
            raise RuntimeError('Please check if state manager of the agent is set correctly.')

    def get_step_info(self):
        """
        Getter method state and action array.
        """

        return self.state, self.action

    def get_carla_api(self, client: carla.Client, tm_port):
        """
        Get carla api. Modified from original get_carla_api function.

        client_timeout is removed, the timeout value is supposed to be set in scenario_runner.

        :param client: carla.Client
        :param tm_port： int, port number of traffic manager
        :return: carla_api dict
        """

        # get current map
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        print("current map for agent is: ", self.map.name)

        debug_helper = self.world.debug  # world debug for plot
        blueprint_library = self.world.get_blueprint_library()  # blueprint
        spectator = self.world.get_spectator()

        # set tm port by args
        traffic_manager = client.get_trafficmanager(tm_port)

        carla_api = {
            'client': client,
            'world': self.world,
            'map': self.map,
            'debug_helper': debug_helper,
            'blueprint_library': blueprint_library,
            'spectator': spectator,
            'traffic_manager': traffic_manager,
        }

        return carla_api

    def set_ego_vehicles(self, ego_vehicle, controller_timestep=0.05):
        """
        Register ego vehicle when episode reset.

        :param controller_timestep:
        :param ego_vehicle: carla.Vehicle
        :return:
        """
        self.ego_vehicle = ego_vehicle
        # controller requires vehicle transform
        self.controller = Controller(self.ego_vehicle, controller_timestep)

    def set_route(self, route):
        """
        todo merge this method with original set global plan

        Setter method.

        Transform route
        """
        self.route = route
        self.reset_buffer()

    def reset_buffer(self):
        """
        Reset waypoint buffer when ego vehicle is reset.

        This method is supposed to be called from outside.
        """

        # todo fix global plan as the route
        # # set waypoint buffer queue
        # for elem in self._global_plan_world_coord:
        #     self._waypoints_queue.append(elem)

        # clear buffered waypoints
        self._waypoints_queue.clear()
        self._waypoint_buffer.clear()

        for elem in self.route:
            self._waypoints_queue.append(elem)

    def buffer_waypoint(self):
        """
        Buffer waypoints for controller.
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
            ego_transform = self.ego_vehicle.get_transform()
            ego_location = ego_transform.location

            # current location of ego vehicle
            # self.ego_location = self.ego_vehicle.get_location()

            # todo check if z coord value effect the dist calculation
            _dist = ego_location.distance(transform.location)

            # todo min distance should be determined by current speed
            _min_dist = 2.  # use a fixed value for convenience

            # if no.i waypoint is in the radius
            if _dist < _min_dist:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):  # (max_index+1) waypoints to pop out of buffer
                self._waypoint_buffer.popleft()

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """

        # todo parse the input data into RL module state vector

        # use state manager to get current state
        state = self.state_manager.get_state()
        self.state = state

        # get ego vehicle from state manager
        self.ego_vehicle = self.state_manager.get_ego_vehicle()

        # ========================================
        # todo get statistics of kinetics, maybe move this to scenario runner
        # todo check if ego vehicle exists
        # ==================================================
        # ----------   get ego vehicle kinetics   ----------
        # ==================================================
        ego_transform = self.ego_vehicle.get_transform()
        # requires transform matrix from world to actor
        trans_matrix = get_transform_matrix(ego_transform)  # actor2world
        inverse_transform_matrix = get_inverse_transform_matrix(ego_transform)  # world2actor

        # kinetics info in ego coord system
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        angular_velocity = self.ego_vehicle.get_angular_velocity()

        # use dict to store
        kinetics = {
            'velocity': velocity,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
        }

        # transform to actor coord system
        transformed_vector = []
        for key in kinetics:
            item = kinetics[key]
            _vec = vector2array(item)
            # trans to actor coord system
            vec = np.dot(inverse_transform_matrix, _vec)
            kinetics[key] = vec

        # todo print timestep info, where to put
        # print('='*20)
        # print action of this timestep
        # print('action: ', action)

        # ========================================

        # ==================================================
        # --------------  Get vehicle control  -------------
        # ==================================================
        self.buffer_waypoint()

        if self._waypoint_buffer:
            # using random policy to debug
            if self.debug:
                action = np.random.rand(2)
                action = list(action)
            else:
                # get an action from the trained model
                action = self.actor.get_action(self.sess, state)

            self.action = action
            target_speed = self.map_action(action)
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

        # transform action to the VehicleControl
        # control.throttle = 0.0

        control = veh_control

        return control

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp))

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    @staticmethod
    def map_action(action):
        """
        Map agent action to the vehicle control command.
        """
        decel_ratio = 0.25
        # map action to vehicle target speed
        target_speed = action[0] - decel_ratio * action[1]
        target_speed = np.clip(target_speed, 0., 1.)
        target_speed = 3.6 * CarlaEnv4.ego_max_speed * target_speed  # in km/h

        return target_speed

    def set_spectator_overhead(self, location, yaw=-90, h=70):
        """
        Set spectator overhead view.

        param location: location of the spectator
        param h(float): height of spectator when using the overhead view
        """

        rotation = carla.Rotation(yaw=yaw, pitch=-90)  # rotate to forward direction
        location.z = h
        spectator = self.carla_api['spectator']
        spectator.set_transform(carla.Transform(location, rotation))
        # self.world.tick()

        print("Spectator is set to overhead view.")

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        ds_ids = downsample_route(global_plan_world_coord, 1)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1])
                                         for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

