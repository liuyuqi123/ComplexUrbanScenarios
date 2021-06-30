"""
A new version of traffic flow manager!

We generate traffic flows whose routes are manually set and controlled by local_planner.

There are 14 available traffic flows in total. The ordering method is that:

 1. start from positive x axis, clockwise direction(+x, +y, -x, -y)

 2. On each direction, turn flag in order of [-1, 0, 1]

           ||   |   ||   |   ||
           ||   |   ||   |   ||
 ----------                    ----------
         -1                     1
          0                     0
          1                    -1
 ----------                    ----------
           ||   |   ||   |   ||
           ||   |   ||   |   ||

----------------   In developing   ----------------

This is a very basic version:

 - considers only 2 traffic flow for right turn task training, negative_x-positive_x, negative_y_0-positive_x

 - all traffic flows are controlled by local planner with hazard detection

"""

import numpy as np
import random
import time

import carla

# todo test using default local planner to control NPC vehicls
# from agents.navigation.basic_agent import LocalPlanner
# from agents.navigation.local_planner import RoadOption

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


def detect_lane_obstacle(world, actor, detect_ego_vehicle=False, extension_factor=3., margin=1.02):
    """
    todo 1.improve this method with an adaptive params, adjust extension factor and margin through current speed
         2.visualize hazard detection

    This method is modified to run without CarlaDataProvider

    This function identifies if an obstacle is present in front of the reference actor.

    :param world
    :param actor: target actor for collision check
    :param detect_ego_vehicle: whether detect ego vehicle when running the
    :param extension_factor: refers to longitudinal detection distance
    :param margin: refers to lateral detection distance
    """
    world_actors = world.get_actors().filter('vehicle.*')

    # if not detecting collision with ego vehicle
    if not detect_ego_vehicle:
        vehicle_list = []
        for veh in world_actors:
            role_name = veh.attributes['role_name']
            if role_name in ['ego', 'hero']:
                continue
            else:
                vehicle_list.append(veh)
    else:
        vehicle_list = world_actors

    actor_bbox = actor.bounding_box
    actor_transform = actor.get_transform()
    actor_location = actor_transform.location
    actor_vector = actor_transform.rotation.get_forward_vector()
    actor_vector = np.array([actor_vector.x, actor_vector.y])
    actor_vector = actor_vector / np.linalg.norm(actor_vector)
    actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
    actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
    actor_yaw = actor_transform.rotation.yaw

    is_hazard = False
    for adversary in vehicle_list:
        if adversary.id != actor.id and \
                actor_transform.location.distance(adversary.get_location()) < 50:
            adversary_bbox = adversary.bounding_box
            adversary_transform = adversary.get_transform()
            adversary_loc = adversary_transform.location
            adversary_yaw = adversary_transform.rotation.yaw
            overlap_adversary = RotatedRectangle(
                adversary_loc.x, adversary_loc.y,
                2 * margin * adversary_bbox.extent.x, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
            overlap_actor = RotatedRectangle(
                actor_location.x, actor_location.y,
                2 * margin * actor_bbox.extent.x * extension_factor, 2 * margin * actor_bbox.extent.y, actor_yaw)
            overlap_area = overlap_adversary.intersection(overlap_actor).area
            if overlap_area > 0:
                is_hazard = True
                break

    return is_hazard


class TrafficFlowManager4(CarlaModule):
    """
    This version of traffic flow manager is supposed to generate a traffic flow
    routed manually and controlled using local_planner.
    """

    # todo add hybrid traffic flows
    # ================   traffic manager params   ================
    tm_params = {
        # percentage to ignore traffic light
        'ignore_lights_percentage': 0.,  # 100.
        # percentage to ignore traffic sign
        'ignore_signs_percentage': 100.,

        # target speed of current traffic flow, in km/h
        # use random.uniform()
        'target_speed': (50., 5),

        # probability refers to whether a vehicle enables collisions(with a target vehicle)
        # the probability to set vehicle collision_detection is True
        'collision_probability': 0.5,  # float [0, 1]

        # lane changing behaviour for a vehicle
        # True is default and enables lane changes. False will disable them.
        'auto_lane_change': False,

        # minimum distance in meters that a vehicle has to keep with the others.
        'distance_to_leading_vehicle': 1.,
    }

    # available traffic flow directions
    tf_directions = ['positive_x', 'negative_x', 'negative_y_0', 'negative_y_1']

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
            'distance_threshold': None,

            'target_speed_range': (15, 25),  # in km/h
            'target_speed': None,
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
            'distance_threshold': None,

            'target_speed_range': (15, 25),  # in km/h
            'target_speed': None,
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
            'distance_threshold': None,

            'target_speed_range': (15, 25),  # in km/h
            'target_speed': None,
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
            'distance_threshold': None,

            'target_speed_range': (15, 25),  # in km/h
            'target_speed': None,
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
            'distance_threshold': None,

            'target_speed_range': (15, 25),  # in km/h
            'target_speed': None,
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
            'distance_threshold': None,

            'target_speed_range': (15, 25),  # in km/h
            'target_speed': None,
        },
    }

    # todo add this into traffic flow info, different tf with different value
    # probability that a single vehicle being enabled to collide with ego vehicle
    collision_probability = 0.75

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
                 ):

        super(TrafficFlowManager4, self).__init__(carla_api=carla_api)

        self.ego_vehicle = None
        self.npc_vehicles = []

        # todo add method to switch active traffic flow
        # set active traffic flow by route option
        self.route_option = route_option

        settings = self.traffic_flow_settings[self.route_option]
        setting_list = zip(settings['tf_directions'], settings['turn_flags'])
        # init with default active tf
        self.active_tf_directions = settings['tf_directions']

        self.turn_flags = {}
        for tf, flag in setting_list:
            self.turn_flags[tf] = flag

        # initialized with default active traffic flow
        self.init_tf_info()

        # # ========   traffic flow   ========
        # self.active_tf_directions = [
        #     # 'positive_x',
        #     'negative_x',
        #     'negative_y_0',
        #     # 'negative_y_1',
        # ]
        # self.turn_flags = {
        #     'negative_x': 0,
        #     'negative_y_0': -1,  # -1, 0
        #     # 'negative_y_1': 1,
        # }

        # ================   Data storage   ================
        # vehicle instance as dict key
        # # dict to store collision sensor
        # self.collision_sensor_dict = {}
        # # dict to store local_planner
        # self.local_planner_dict = {}

        self.block_time_dict = {}

        # todo add a list to store all info dict, make clear more convenient
        # if vehicle has passed the junction
        self.enter_junction_flag_dict = {}
        self.pass_junction_flag_dict = {}

        # get dt from world settings
        settings = self.world.get_settings()
        self.simulation_timestep = settings.fixed_delta_seconds

        # junction info
        self.junction_edges = {}
        self.get_junction_info()

        self.debug = True

    def init_tf_info(self):
        """
        Init traffic flow according to active traffic flow.

        Init following info:
        - route
        - sink location
        """
        for tf in self.active_tf_directions:
            # retrieve info dict
            info_dict = self.traffic_flow_info[tf]
            spawn_transform = info_dict['spawn_transform']
            turn_flag = self.turn_flags[tf]

            route = self.get_route(spawn_transform,
                                   turn_flag=turn_flag,
                                   hop_resolution=1.,
                                   )

            # todo append route to dict
            self.traffic_flow_info[tf]['route'] = route

            # get sink location
            sink_location = route[-5][0].location
            self.traffic_flow_info[tf]['sink_location'] = sink_location

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

    def run_step(self):
        """
        Tick the traffic flow
        """

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # spawn new vehicle
        self.actor_source()

        self.tick_local_planners()

        # todo merge the method into one
        # remove collision vehicle
        self.collision_handling()

        # remove blocked vehicle
        self.update_block_time()
        self.remove_blocked_vehicles()

        # check and remove vehicle which reaches the end location
        self.actor_sink()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # print('')

    def update_block_time(self):
        """
        todo add a method to check traffic flow vehicles delete condition

        Update the block time of all active traffic flow vehicles.

        This method is supposed to be called each timestep.
        """

        for vehicle in self.block_time_dict:

            # get current velocity
            speed = get_speed(vehicle)  # in km/h

            if speed <= 1.:  # check by a threshold
                self.block_time_dict[vehicle] += self.simulation_timestep  # todo use timestep length
            else:
                # reset the block time
                self.block_time_dict[vehicle] = 0

    def remove_blocked_vehicles(self, max_block_time=5.):
        """
        Check if a vehicle is blocked,
        usually blocked in junction, or blocked by other traffic lights.

        todo add args to set maximum block time in and out of the junction

        param vehicle: vehicle to be checked
        param max_block_time: the maximum block time for npc vehicle
                              which is outside the junction

        return: bool, True refers that the vehicle is blocked
        """
        delete_list = []

        # for tf in self.active_tf_directions:
        # fixme check all tf directions
        for tf in self.traffic_flow_info:

            info_dict = self.traffic_flow_info[tf]
            vehicle_list = info_dict['vehicle_list']
            sensor_dict = info_dict['sensor_dict']

            for vehicle in vehicle_list:
                block_time = self.block_time_dict[vehicle]
                if block_time >= max_block_time:
                    sensor = sensor_dict[vehicle]

                    delete_list.append(vehicle)
                    delete_list.append(sensor)

                    # todo put this into a method
                    # remove from data storage
                    self.traffic_flow_info[tf]['vehicle_list'].remove(vehicle)  # list
                    self.traffic_flow_info[tf]['sensor_dict'].pop(vehicle)  # dict
                    self.traffic_flow_info[tf]['sensor_api_dict'].pop(vehicle)  # dict
                    self.traffic_flow_info[tf]['local_planner_dict'].pop(vehicle)
                    self.traffic_flow_info[tf]['collision_flag_dict'].pop(vehicle)
                    self.block_time_dict.pop(vehicle)

        # remove actors
        self.delete_actors(delete_list)

    def collision_handling(self):
        """
        Remove collision vehicles and sensors
        """
        delete_list = []

        # for tf in self.active_tf_directions:
        # fixme check all tf directions
        for tf in self.traffic_flow_info:

            info_dict = self.traffic_flow_info[tf]
            vehicle_list = info_dict['vehicle_list']
            sensor_api_dict = info_dict['sensor_api_dict']
            sensor_dict = info_dict['sensor_dict']

            for vehicle in vehicle_list:
                sensor_api = sensor_api_dict[vehicle]
                collision_flag = sensor_api.collision_flag
                if collision_flag:
                    sensor = sensor_dict[vehicle]

                    delete_list.append(vehicle)
                    delete_list.append(sensor)

                    # todo put this into a method
                    # remove from data storage
                    self.traffic_flow_info[tf]['vehicle_list'].remove(vehicle)  # list
                    self.traffic_flow_info[tf]['sensor_dict'].pop(vehicle)  # dict
                    self.traffic_flow_info[tf]['sensor_api_dict'].pop(vehicle)  # dict
                    self.traffic_flow_info[tf]['local_planner_dict'].pop(vehicle)
                    self.traffic_flow_info[tf]['collision_flag_dict'].pop(vehicle)
                    self.block_time_dict.pop(vehicle)

                    self.enter_junction_flag_dict.pop(vehicle)
                    self.pass_junction_flag_dict.pop(vehicle)

        # remove actors
        self.delete_actors(delete_list)

    def actor_source(self):
        """
        This method is supposed to be called
        """
        for tf in self.active_tf_directions:

            # retrieve information from info dict
            info_dict = self.traffic_flow_info[tf]

            # existing vehicles of current traffic flow
            vehicle_list = []
            for veh in info_dict['vehicle_list']:
                vehicle_list.append(veh)

            # todo check if need more
            # check distance condition
            spawn_transform = info_dict['spawn_transform']
            # get center point of lane
            spawn_waypoint = self.map.get_waypoint(spawn_transform.location,
                                                   project_to_road=True,  # not in the center of lane(road)
                                                   lane_type=carla.LaneType.Driving,
                                                   )
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 0.2

            # for the first vehicle in the
            if not info_dict['distance_threshold']:  # for initializing
                distance_threshold = info_dict['distance_range'][0] + 5.  # 5.0 is vehicle length
            else:
                distance_threshold = info_dict['distance_threshold'] + 5.

            # only check current traffic flow
            distance_condition = False
            if vehicle_list:
                last_spawn_vehicle = vehicle_list[-1]
                if spawn_transform.location.distance(last_spawn_vehicle.get_location()) >= distance_threshold:
                    distance_condition = True
            else:
                distance_condition = True

            # if not spawning new vehicle, continue
            if not distance_condition:
                return
            else:
                # update distance_threshold for next vehicle
                distance_lower_limit, distance_upper_limit = info_dict['distance_range']
                new_distance_threshold = random.uniform(distance_lower_limit, distance_upper_limit)
                self.traffic_flow_info[tf]['distance_threshold'] = new_distance_threshold

            # spawn new vehicle
            try:
                # todo fix usage of info_dict
                # spawn new actor
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

            # init local planner with params
            # set target speed of the vehicle
            speed_lower_limit, speed_upper_limit = info_dict['target_speed_range']
            target_speed = random.uniform(speed_lower_limit, speed_upper_limit)
            target_speed = target_speed

            # todo improve this part
            """
            An new actor will not be shown in world until world.tick() is ran,
            The transform is initialized with all zero, 
            which will lead to a wrong velocity direction.
            A fix method is to use spawn transform to calculate velocity direction.              
            """
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
                    'K_D': 0,
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

            # append vehicle info dicts
            self.traffic_flow_info[tf]['vehicle_list'].append(vehicle)
            self.traffic_flow_info[tf]['sensor_api_dict'][vehicle] = sensor
            self.traffic_flow_info[tf]['sensor_dict'][vehicle] = sensor.collision
            self.traffic_flow_info[tf]['local_planner_dict'][vehicle] = local_planner
            self.traffic_flow_info[tf]['collision_flag_dict'][vehicle] = collision_detection

            # ================   Init vehicle info dict   ================
            # init block time
            self.block_time_dict[vehicle] = 0.

            # init pass junction flag
            self.enter_junction_flag_dict[vehicle] = False
            self.pass_junction_flag_dict[vehicle] = False

            # print('New vehicle spawned by ActorSource')

    def get_junction_info(self):
        """
        Get info of junction
        :return:
        """
        # todo add api to get junction
        # get default junction
        junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)
        wp = self.map.get_waypoint(location=junction_center,
                                   project_to_road=False,  # not in the center of lane(road)
                                   lane_type=carla.LaneType.Driving)
        junction = wp.get_junction()
        # junction edges' coordinates
        self.junction_edges = self.get_junction_edges(junction, margin=0.)

    @staticmethod
    def get_junction_edges(junction, margin=0.):
        """
        Parse junction info.
        """
        margin = np.clip(margin, 0., 99.)

        bbox = junction.bounding_box
        extent = bbox.extent
        location = bbox.location

        # todo fix for junction whose bbox has a rotation
        # rotation = bbox.rotation  # original rotation of the bbox

        # get junction vertices
        x_max = location.x + extent.x + margin
        x_min = location.x - extent.x - margin
        y_max = location.y + extent.y + margin
        y_min = location.y - extent.y - margin

        junction_edges = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
        }

        return junction_edges

    def check_pass_junction(self, vehicle):
        """
        Check if vehicle pass the junction area.

        todo if need to move this method into run_step
        """
        # edges of junction
        x_min = self.junction_edges['x_min']
        x_max = self.junction_edges['x_max']
        y_min = self.junction_edges['y_min']
        y_max = self.junction_edges['y_max']
        #
        location = vehicle.get_location()
        x, y = location.x, location.y

        # current state
        in_junction = x_min <= x <= x_max and y_min <= y <= y_max

        # previous state
        enter_flag = self.enter_junction_flag_dict[vehicle]

        # check if pass the junction
        if not enter_flag:
            if in_junction:
                self.enter_junction_flag_dict[vehicle] = True  # update enter_flag

        pass_flag = False
        if enter_flag:
            if not in_junction:
                pass_flag = True

        return pass_flag

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

                # todo fix with a better api
                if tf == 'negative_y_0':
                    extension_factor = 3.
                    margin = 1.05
                else:
                    extension_factor = 3.
                    margin = 1.05

                # get safety action overwrite original action
                if detect_lane_obstacle(self.world,
                                        vehicle,
                                        detect_ego_vehicle=avoid_collision,
                                        extension_factor=extension_factor,
                                        margin=margin,
                                        ):
                    control.throttle = 0.0
                    control.brake = 1.0

                vehicle.apply_control(control)

    def actor_sink(self):
        """
        Delete vehicle when arriving end of the route.

        end_location indicates the end of the route.
        """
        delete_list = []

        # for tf in self.active_tf_directions:
        # fixme check all tf directions
        for tf in self.traffic_flow_info:

            info_dict = self.traffic_flow_info[tf]

            # get sink location from dict
            sink_location = info_dict['sink_location']
            for vehicle in info_dict['vehicle_list']:
                location = vehicle.get_location()
                sink_threshold = 1.5  # lane width is 3.5
                if sink_location.distance(location) < sink_threshold:
                    sensor = info_dict['sensor_dict'][vehicle]
                    delete_list.append(vehicle)
                    delete_list.append(sensor)

                    # remove from all data storage
                    self.traffic_flow_info[tf]['vehicle_list'].remove(vehicle)  # list
                    self.traffic_flow_info[tf]['sensor_dict'].pop(vehicle)  # dict
                    self.traffic_flow_info[tf]['sensor_api_dict'].pop(vehicle)  # dict
                    self.traffic_flow_info[tf]['local_planner_dict'].pop(vehicle)
                    self.traffic_flow_info[tf]['collision_flag_dict'].pop(vehicle)
                    self.block_time_dict.pop(vehicle)

                    self.enter_junction_flag_dict.pop(vehicle)
                    self.pass_junction_flag_dict.pop(vehicle)

        # remove actors
        self.delete_actors(delete_list)

    # ================   Developed Methods   ================

    def clean_up(self):
        """
        Clean all vehicles of traffic flow.
        """
        delete_list = []

        for tf in self.active_tf_directions:
            info_dict = self.traffic_flow_info[tf]
            for vehicle in info_dict['vehicle_list']:
                sensor = info_dict['sensor_dict'][vehicle]
                delete_list.append(vehicle)
                delete_list.append(sensor)

                # todo check this part
                # remove from all data storage
                self.traffic_flow_info[tf]['vehicle_list'].clear()  # list
                self.traffic_flow_info[tf]['sensor_dict'].clear()  # dict
                self.traffic_flow_info[tf]['sensor_api_dict'].clear()  # dict
                self.traffic_flow_info[tf]['local_planner_dict'].clear()
                self.traffic_flow_info[tf]['collision_flag_dict'].clear()
                self.block_time_dict.clear()

                self.enter_junction_flag_dict.clear()
                self.pass_junction_flag_dict.clear()

        # remove actors
        self.delete_actors(delete_list)

    def delete_actors(self, delete_list):
        """
        Delete actors stored in list.
        """
        actor_id = [x.id for x in delete_list]

        # carla.Client.apply_batch_sync method
        response_list = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in delete_list], False)

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

    @staticmethod
    def set_velocity(vehicle, transform, target_speed: float):
        """
        Set a vehicle to the target velocity.

        params: target_speed: in m/s
        """
        # if using spawn transform
        # transform = vehicle.get_transform()

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
        #     self.world.tick()

    def spawn_single_vehicle(self, transform, name=None):
        """
        Spawn single NPC vehicle at given transform.

        :return: vehicle of carla.Vehicle
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        # bp = self.blueprint_library.filter('vehicle.lincoln.mkz2017')

        if bp.has_attribute('color'):
            color = '0, 255, 255'  # use string to identify a RGB color
            bp.set_attribute('color', color)
        # use sticky control
        # bp.set_attribute('sticky_control', 'False')

        if name:
            bp.set_attribute('role_name', name)  # set actor name
        # spawn npc vehicle
        try:
            vehicle = self.world.spawn_actor(bp, transform)  # use spawn method

            # necessary to tick world, seem not necessary to sleep
            # self.world.tick()
            # time.sleep(0.05)
        except:
            raise RuntimeError('Fail to spawn a NPC vehicle, please check.')

        return vehicle

    def get_route(self, spawn_transform, turn_flag, hop_resolution=1.):
        """
        Get route for single traffic flow.

        :params: spawn_transform:
        :params: turn_flag:

        return: route in transform tuple
        """
        spawn_waypoint = self.map.get_waypoint(spawn_transform.location,
                                               project_to_road=True,
                                               )
        # start waypoint of the route
        gap_distance = 3.
        start_waypoint = spawn_waypoint.next(gap_distance)[0]
        start_location = start_waypoint.transform.location

        # exit waypoint of the junction
        exit_waypoint = generate_target_waypoint(start_waypoint, turn_flag)
        exit_location = exit_waypoint.transform.location

        # fixed method to get end location
        max_distance_after_junction = 55
        traveled_distance = 0
        end_waypoint = exit_waypoint.next(1.0)[0]
        while not end_waypoint.is_intersection and traveled_distance < max_distance_after_junction:
            waypoint_new = end_waypoint.next(1.)[-1]
            traveled_distance += waypoint_new.transform.location.distance(end_waypoint.transform.location)
            end_waypoint = waypoint_new

        end_location = end_waypoint.transform.location

        # # end_waypoint refers to the last waypoint till next intersection
        # end_waypoint = exit_waypoint.next(1.0)[0]
        # while not end_waypoint.is_intersection:
        #     end_waypoint = end_waypoint.next(1.0)[0]  # end_waypoint refers to the end of the whole route
        # end_location = end_waypoint.transform.location

        location_list = [
            start_location,
            exit_location,
            end_location,
        ]

        # route of transform tuple
        _, trans_route = interpolate_trajectory(world=self.world,
                                                waypoints_trajectory=location_list,
                                                hop_resolution=hop_resolution,
                                                )

        for trans, _ in trans_route:
            draw_waypoint(self.world, trans, color=(magenta, magenta))

        return trans_route

    def set_collision_probability(self, collision_probability):
        """
        Setter method for collision_probability.
        """
        self.collision_probability = collision_probability
