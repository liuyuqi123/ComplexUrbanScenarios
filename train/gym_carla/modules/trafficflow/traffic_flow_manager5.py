"""
This method is a developing version of traffic manager4.

Major modification:

 - add OU noise for traffic flow velocity and distance range exploration.

 - assign a noise generator for each traffic flow

 - store params of the generated vehicles

 - add a class attribute(self.randomize) to switch on/off noise generator of the target speed and distance

todo:
 - add a arg to set if enables collision detection with ego vehicle

"""

import numpy as np
import random
import time

import carla

from train.gym_carla.navigation.local_planner import LocalPlanner

from train.gym_carla.modules.carla_module import CarlaModule

from train.gym_carla.util_development.sensors import Sensors
from train.gym_carla.util_development.kinetics import get_transform_matrix
from train.gym_carla.util_development.scenario_helper_modified import generate_target_waypoint
from train.gym_carla.util_development.route_manipulation import interpolate_trajectory

from train.gym_carla.util_development.scenario_helper_modified import RotatedRectangle
from train.gym_carla.util_development.util_visualization import draw_waypoint
from train.gym_carla.util_development.carla_color import *
from train.gym_carla.navigation.misc import get_speed

from train.gym_carla.modules.trafficflow.traffic_flow_manager4 import TrafficFlowManager4, detect_lane_obstacle
from train.gym_carla.modules.trafficflow.ou_noise import (TruncatedOUNoise, CoordinateNoise)


class TrafficFlowManager5(TrafficFlowManager4):
    # todo current route route info is for Town 03 junction, add API methods for random junction
    # todo use a seperate file to store the param setting
    # dict to store traffic flow information
    traffic_flow_info = {
        'positive_x': {
            'spawn_transform': carla.Transform(carla.Location(x=71.354889, y=130.074112, z=0.018447),
                                               carla.Rotation(pitch=359.836853, yaw=179.182800, roll=0.000000)),
            'sink_location': None,

            'route': None,  # route or plan, waypoints tuple
            'vehicle_list': [],

            'sensor_api_dict': {},
            'sensor_dict': {},
            'local_planner_dict': {},
            'collision_flag_dict': {},

            'distance_range': (10, 40),  # in meters
            'distance_threshold': 25,

            'target_speed_range': (10, 45),  # in km/h
            'target_speed': 25,
        },

        'negative_x': {
            'spawn_transform': carla.Transform(carla.Location(x=-64.222389, y=135.423065, z=0.200000),
                                               carla.Rotation(pitch=0.000000, yaw=-361.296783, roll=0.000000)),
            'sink_location': None,

            'route': None,  # route or plan, waypoints tuple
            'vehicle_list': [],

            'sensor_api_dict': {},
            'sensor_dict': {},
            'local_planner_dict': {},
            'collision_flag_dict': {},

            'distance_range': (10, 40),  # in meters
            'distance_threshold': 25,

            'target_speed_range': (10, 45),  # in km/h
            'target_speed': 25,
        },

        'negative_y_0': {
            'spawn_transform': carla.Transform(carla.Location(x=-6.411462, y=68.223877, z=0.100000),
                                               carla.Rotation(pitch=0.000000, yaw=89.637459, roll=0.000000)),
            'sink_location': None,

            'route': None,  # route or plan, waypoints tuple
            'vehicle_list': [],

            'sensor_api_dict': {},
            'sensor_dict': {},
            'local_planner_dict': {},
            'collision_flag_dict': {},

            'distance_range': (10, 40),  # in meters
            'distance_threshold': 25,

            'target_speed_range': (10, 45),  # in km/h
            'target_speed': 25,
        },

        'negative_y_1': {
            'spawn_transform': carla.Transform(carla.Location(x=-9.911532, y=68.223877, z=0.200000),
                                               carla.Rotation(pitch=0.000000, yaw=89.637459, roll=0.000000)),
            'sink_location': None,

            'route': None,  # route or plan, waypoints tuple
            'vehicle_list': [],

            'sensor_api_dict': {},
            'sensor_dict': {},
            'local_planner_dict': {},
            'collision_flag_dict': {},

            'distance_range': (10, 40),  # in meters
            'distance_threshold': 25,

            'target_speed_range': (10, 45),  # in km/h
            'target_speed': 25,
        },

        # ego vehicle spawned on positive y direction
        # positive y direction is not used usually
        'positive_y_0': {
            'spawn_transform': carla.Transform(carla.Location(x=2.354240, y=189.210159, z=0.200000),
                                               carla.Rotation(pitch=0.000000, yaw=-90.362534, roll=0.000000)),
            'sink_location': None,

            'route': None,  # route or plan, waypoints tuple
            'vehicle_list': [],

            'sensor_api_dict': {},
            'sensor_dict': {},
            'local_planner_dict': {},
            'collision_flag_dict': {},

            'distance_range': (10, 40),  # in meters
            'distance_threshold': 25,

            'target_speed_range': (10, 45),  # in km/h
            'target_speed': 25,
        },

        'positive_y_1': {
            'spawn_transform': carla.Transform(carla.Location(x=5.854938, y=189.309280, z=0.200000),
                                               carla.Rotation(pitch=0.000000, yaw=-90.362534, roll=0.000000)),
            'sink_location': None,

            'route': None,  # route or plan, waypoints tuple
            'vehicle_list': [],

            'sensor_api_dict': {},
            'sensor_dict': {},
            'local_planner_dict': {},
            'collision_flag_dict': {},

            'distance_range': (10, 40),  # in meters
            'distance_threshold': 25,

            'target_speed_range': (10, 45),  # in km/h
            'target_speed': 25,
        },
    }

    # traffic flow direction settings
    traffic_flow_settings = {
        'left': {
            'tf_directions': ['negative_y_0', 'negative_y_1'],
            'turn_flags': [0, 1],
        },

        'right': {
            'tf_directions': ['negative_x'],
            'turn_flags': [0],
        },

        'straight_0': {
            'tf_directions': ['negative_x', 'negative_y_0'],
            'turn_flags': [0, -1],

            # ============  debug  ============
            # # only negative_x straight
            # 'tf_directions': ['negative_x'],
            # 'turn_flags': [0],

            # # only negative_y_0 left
            # 'tf_directions': ['negative_y_0'],
            # 'turn_flags': [-1],
        },

        # this training is not included into scenario test
        'straight_1': {
            'tf_directions': ['negative_x', 'negative_y_0'],
            'turn_flags': [0, -1],
        },
    }

    def __init__(self,
                 carla_api,
                 route_option,
                 randomize=True,  # whether use noise generator
                 ):

        super(TrafficFlowManager5, self).__init__(carla_api, route_option)

        # store params of generated vehicles
        self.tf_distribution = {}

        # init noise generator
        self.speed_generators = {}
        self.distance_generators = {}
        self.init_noise_generator()

        self.randomize = randomize

        # if switch off the randomization of traffic flow params
        if not self.randomize:
            for tf in self.traffic_flow_info:
                # retrieve lower limit
                distance_lower_limit = self.traffic_flow_info[tf]['distance_range'][0]
                self.traffic_flow_info[tf]['distance_threshold'] = distance_lower_limit

                speed_lower_limit = self.traffic_flow_info[tf]['target_speed_range'][0]
                self.traffic_flow_info[tf]['target_speed'] = speed_lower_limit

    def init_noise_generator(self):
        """
        todo current method only init generator and recorder for default active traffic flow

        Init noise generator for each traffic flow.

        We deploy a stochastic process to generate target velocity and distance gap.

        Different traffic flow doesn't share noise generator because the params maybe various.
        """
        for tf in self.active_tf_directions:
            # retrieve information
            info_dict = self.traffic_flow_info[tf]
            speed_range = info_dict['target_speed_range']
            # speed generator
            speed_generator = TruncatedOUNoise(
                a=speed_range[0],
                b=speed_range[1],
                n_sigma=1.9,
                theta=0.24,
            )
            self.speed_generators[tf] = speed_generator

            # distance
            distance_range = info_dict['distance_range']
            distance_generator = CoordinateNoise(
                source_range=speed_range,
                target_range=distance_range,
                n_sigma=1.2,  # proportion is near 1:3
            )
            self.distance_generators[tf] = distance_generator

            # use a list to store traffic flow params
            # element format: (target_speed, distance)
            # distance refers to the distance in meters relative to previous vehicle
            self.tf_distribution[tf] = []

    def actor_source(self):
        """
        This method is supposed to be called
        """
        for tf in self.active_tf_directions:
            # retrieve information from info dict
            info_dict = self.traffic_flow_info[tf]

            # get existing vehicles of current traffic flow
            vehicle_list = []
            for veh in info_dict['vehicle_list']:
                vehicle_list.append(veh)

            spawn_transform = info_dict['spawn_transform']
            distance_threshold = info_dict['distance_threshold'] + 5.

            # check distance condition
            distance_condition = False
            if vehicle_list:
                last_spawn_vehicle = vehicle_list[-1]
                if spawn_transform.location.distance(last_spawn_vehicle.get_location()) >= distance_threshold:
                    distance_condition = True
            else:  # if vehicle list is None, there are no vehicles in current traffic flow
                distance_condition = True

            # if not spawning new vehicle, continue
            if not distance_condition:
                return

            # spawn new vehicle
            try:
                # get waypoint at the center of lane
                spawn_waypoint = self.map.get_waypoint(spawn_transform.location,
                                                       project_to_road=True,  # not in the center of lane(road)
                                                       lane_type=carla.LaneType.Driving,
                                                       )
                spawn_transform = spawn_waypoint.transform
                spawn_transform.location.z += 0.2
                vehicle = self.spawn_single_vehicle(spawn_transform, name=tf)
            except:
                print("Fail to spawn new actor in tf: ", tf)
                return

            # set collision detection with a probability
            if np.random.random() <= self.collision_probability:
                collision_detection = False  # refers to enable collision
            else:
                collision_detection = True

            # create collision sensor
            sensor = Sensors(self.world, vehicle)

            # # todo add try tick carla method
            # self.world.tick(100.)

            # todo improve this part
            """
            An new actor will not be shown in world until world.tick() is ran,
            The transform is initialized with all zero, 
            which will lead to a wrong velocity direction.
            A fix method is to use spawn transform to calculate velocity direction.              
            """

            # init local planner with params
            target_speed = info_dict['target_speed']
            # set initial velocity
            self.set_velocity(vehicle, spawn_transform, 0.95 * target_speed / 3.6)  # carla set velocity takes m/s

            # opt dict to init local planner
            opt_dict = {
                'target_speed': target_speed,
                'dt': self.simulation_timestep,
                'lateral_control_dict': {
                    'K_P': 1.95,
                    'K_D': 0.2,
                    'K_I': 0.07,
                    'dt': self.simulation_timestep,
                },
                'longitudinal_control_dict': {
                    'K_P': 1.0,
                    'K_D': 0.,
                    'K_I': 0.05,
                    'dt': self.simulation_timestep,
                }
            }
            # initialize waypoint follower
            local_planner = LocalPlanner(vehicle, carla_map=self.map, opt_dict=opt_dict)

            # set route for the vehicle
            route = self.traffic_flow_info[tf]['route']
            # todo check route elements
            local_planner.set_global_plan(route)

            # ================   update params for next vehicle   ================
            # store tf param of this vehicle
            self.tf_distribution[tf].append((target_speed, distance_threshold))

            if self.randomize:
                # generate distance gap of next vehicle based on previous vehicle speed
                distance_generator = self.distance_generators[tf]
                new_distance_threshold = distance_generator(target_speed)
                self.traffic_flow_info[tf]['distance_threshold'] = new_distance_threshold

                # target velocity
                speed_generator = self.speed_generators[tf]
                new_target_speed = speed_generator()[0]  # todo return is ndarray, if fix the noise class
                self.traffic_flow_info[tf]['target_speed'] = new_target_speed

            # ================   update info storage   ================
            # append vehicle info dicts
            self.traffic_flow_info[tf]['vehicle_list'].append(vehicle)
            self.traffic_flow_info[tf]['sensor_api_dict'][vehicle] = sensor
            self.traffic_flow_info[tf]['sensor_dict'][vehicle] = sensor.collision
            self.traffic_flow_info[tf]['local_planner_dict'][vehicle] = local_planner
            self.traffic_flow_info[tf]['collision_flag_dict'][vehicle] = collision_detection

            # init block time
            self.block_time_dict[vehicle] = 0.

            # init pass junction flag
            self.enter_junction_flag_dict[vehicle] = False
            self.pass_junction_flag_dict[vehicle] = False

            # print('New vehicle spawned by ActorSource')

    def tick_local_planners(self):
        """
        Tick all vehicles controlled by local planner.
        """
        # for tf in self.active_tf_directions:
        # fixme check all tf directions
        for tf in self.traffic_flow_info:

            for vehicle in self.traffic_flow_info[tf]['vehicle_list']:

                # todo test this line
                # retrieve object from info dict
                local_planner = self.traffic_flow_info[tf]['local_planner_dict'][vehicle]

                # flag to identify whether this vehicle avoid collision with ego vehicle
                avoid_collision = self.traffic_flow_info[tf]['collision_flag_dict'][vehicle]

                # retrieve previous pass flag
                pass_flag = self.pass_junction_flag_dict[vehicle]
                # update if has not passed yet
                if not pass_flag:
                    # check if vehicle has passed the junction
                    pass_flag = self.check_pass_junction(vehicle)

                    if pass_flag:
                        self.pass_junction_flag_dict[vehicle] = True
                        # set vehicle to a higher target velocity if vehicle has exited the junction
                        local_planner.set_speed(40)  # default max velocity, in m/s

                # this line includes buffering waypoints
                control = local_planner.run_step(debug=False)

                # =============  Detect potential collision  =============
                # todo improve this API for different traffic flow
                # set larger collision detection area for the left turning traffic flow
                if tf == 'negative_y_0':
                    extension_factor = 2.75
                    margin = 1.05
                else:
                    extension_factor = 3.
                    margin = 1.05

                # # todo the extension_factor refers to longitudinal range, should be set according to vehicle speed
                # extension_factor = 3.
                # margin = 1.05

                # todo visualize collision detection method
                # get safety action overwrite original action
                if detect_lane_obstacle(self.world,
                                        vehicle,
                                        detect_ego_vehicle=avoid_collision,
                                        extension_factor=extension_factor,
                                        margin=margin,
                                        ):
                    # replace original longitudinal action with full braking
                    control.throttle = 0.0
                    control.brake = 1.0

                vehicle.apply_control(control)
