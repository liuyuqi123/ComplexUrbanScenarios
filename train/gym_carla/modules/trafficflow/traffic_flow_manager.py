"""
This is a improved version of traffic flow module.

The manager is responsible for generate and maintain a traffic flow.

New functions:

 - add api to set npc vehicles' behavior

 - set active traffic flow manually

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

import numpy as np
import random

from train.gym_carla.util_development.scenario_helper_modified import generate_target_waypoint
from train.gym_carla.util_development.util_junction import (plot_coordinate_frame)
from train.gym_carla.util_development.util_visualization import draw_waypoint
from train.gym_carla.navigation.misc import get_speed

from train.gym_carla.util_development.sensors import Sensors


# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)

# todo move config info to a file
# default info
start_location = carla.Location(x=53.0, y=128.0, z=3.0)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)

# default Town03 junction center location
junction_center_location = carla.Location(x=-1.32, y=132.69, z=0.00)


class TrafficFlowManager:
    """
    This class is responsible for generation and management
     of traffic flow in a intersection scenario.
    """

    # available traffic flow directions
    tf_directions = ['positive_x', 'negative_x', 'positive_y', 'negative_y']

    # params for traffic manager
    tm_params = {

        # percentage to ignore traffic light
        'ignore_lights_percentage': 0.,  # 100.
        # percentage to ignore traffic sign
        'ignore_signs_percentage': 100.,

        # target speed of current traffic flow, in km/h
        'target_speed': 50.,

        # probability refers to whether a vehicle enables collisions(with a target vehicle)
        # the probability to set vehicle collision_detection is True
        'collision_probability': 0.99,  # float [0, 1]
    }

    # time interval to spawn new vehicles
    lower_limit = 1.0
    upper_limit = 10.0
    # min distance to last spawned vehicle
    distance_threshold = 5.0

    # todo 1. add setter or args to set value
    #  2. set different max-block time for different direction??
    # max block time is supposed to be larger than
    # the duration of red state phase of traffic light(y-direction)
    red_phase_time = 20.

    # todo use probability to spawn npc vehicles, ref on sumo method
    # prob_dict = {
    #     'left': 0.11,
    #     'mid': 0.11,
    #     'right': 0.11,
    # }

    # todo the params are for Town 03 junction, add method to suit random junction
    # this dict stores all traffic flow information
    traffic_flow_info = {
        'positive_x': {
            'spawn_transform': carla.Transform(carla.Location(x=53.0, y=128.0, z=0.000000),
                                               carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],  # list of tuple, (vehicle, sensor)
        },

        'positive_y': {
            'spawn_transform': carla.Transform(carla.Location(x=5.767616, y=175.509048, z=0.000000),
                                               carla.Rotation(pitch=0.0, yaw=269.637451, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'negative_y': {
            'spawn_transform': carla.Transform(carla.Location(x=-6.268355, y=90.840492, z=0.000000),
                                               carla.Rotation(pitch=0.0, yaw=89.637459, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'negative_x': {
            'spawn_transform': carla.Transform(carla.Location(x=-46.925552, y=135.031494, z=0.000000),
                                               carla.Rotation(pitch=0.0, yaw=-1.296802, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        }
    }

    def __init__(self,
                 carla_api,
                 junction=None,
                 active_tf_direction=None,
                 tm_seed=int(0),
                 debug=False,
                 ):
        """
        Init the manager.

        params: active_tf_direction: the active traffic flow directions
        """

        self.carla_api = carla_api
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']
        # traffic manager port number
        self.tm_port = self.traffic_manager.get_port()

        # set traffic manager seed
        self.traffic_manager.set_random_device_seed(tm_seed)

        # get timestep length of simulator
        settings = self.world.get_settings()
        self.timestep_length = settings.fixed_delta_seconds

        # assign junction
        if junction:
            self.junction = junction
        else:
            self.junction = self.get_junction_by_location(self.map, junction_center_location)

        # active_tf_direction determine which traffic flow is activated
        if active_tf_direction:
            self.active_tf_direction = active_tf_direction
        else:
            self.active_tf_direction = list(self.traffic_flow_info.keys())

        self.set_active_tf_direction(self.active_tf_direction, clean_vehicles=False)

        # vehicles
        self.ego_vehicle = None
        self.npc_vehicles = []
        # list to store vehicle and its info_dict, element: (vehicle, vehicle_info)
        # format of vehicle info dict
        self.vehicle_info_list = []
        """
        info_dict = {
            'id': vehicle.id,
            'vehicle': vehicle,  # carla.Vehicle
            'sensor_api': sensor,  # instance of Sensor
            'sensor': sensor.collision,  # carla.Actor
            'block_time': 0.,  # the time this vehicle has been blocked
        }    
        """

        # debug mode
        self.debug = debug

    def __call__(self):
        """
        todo may not necessary.

        if some methods require multiple called
        """
        self.run_step()

    def run_step(self):
        """
        Tick the traffic flow at each timestep.

        # todo print newly spawned and deleted vehicles.
        """
        # try to spawn new traffic flow vehicles
        self.spawn_new_vehicles()
        # check and delete npc vehicles
        self.delete_npc()
        # make sure existing npc vehicles be aware of ego vehicle
        self.update_vehicles()

        if self.ego_vehicle:
            for veh in self.npc_vehicles:
                self.set_collision_detection(veh, self.ego_vehicle)

    def update_vehicles(self):
        """
        Check and update vehicles exist in current world.
        """
        # reset the storage
        npc_vehicles = []
        self.ego_vehicle = None

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a actorlist instance, iterable

        if vehicle_list:
            for veh in vehicle_list:
                attr = veh.attributes  # dict
                # filter ego vehicle by role name
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':
                    self.ego_vehicle = veh
                else:
                    npc_vehicles.append(veh)

        # check if npc vehicles are right
        id_list = []
        _id_list = []

        for veh in npc_vehicles:
            id_list.append(veh.id)

        for veh in self.npc_vehicles:
            _id_list.append(veh.id)

        if self.debug:
            _difference = set(id_list) ^ set(_id_list)
            if _difference:
                print('These vehicles are not updated correctly: ')
                print(_difference)

    def set_active_tf_direction(self, active_tf_direction: list, clean_vehicles=False):
        """
        A setter method,
        Set traffic flow direction.

        tf_direction refers to which direction of traffic flow is activated.
        Optional tf_directions:
         - positive_x
         - positive_y
         - negative_x
         - negative_y

        params: tf_direction: list contains desired traffic flow direction, i.e. [positive_x, positive_y]
        params: clean_vehicles:  if remove vehicles from inactive traffic flows
        """
        # check and update tf_direction
        for tf_direc in active_tf_direction:
            if tf_direc not in self.traffic_flow_info.keys():
                raise ValueError('Wrong traffic flow direction, please check.')
        self.active_tf_direction = active_tf_direction

        # vehicles in inactive traffic flow will be removed
        if clean_vehicles:
            # tf directions to be cleaned
            tf_direction_to_remove = [tf for tf in self.tf_directions if tf not in self.active_tf_direction]

            delete_actors = []  # actors to delete
            # get actors to be removed
            for tf in tf_direction_to_remove:
                _vehicle_sensor_list = self.traffic_flow_info[tf]['vehicle_sensor']
                for item in _vehicle_sensor_list:
                    vehicle = item[0]
                    sensor = item[1]
                    delete_actors.append(vehicle)  # vehicle
                    delete_actors.append(sensor)  # collision sensor actor

                    # remove from class attrubutes
                    for veh in self.npc_vehicles:
                        if veh.id == item[0]:
                            self.npc_vehicles.remove(veh)

                    for tup in self.vehicle_info_list:
                        _veh = tup[0]
                        if _veh.id == vehicle.id:
                            self.vehicle_info_list.remove(tup)

                # reset the tf dict in npc_info
                self.traffic_flow_info[tf]['last_spawn_time'] = 0
                self.traffic_flow_info[tf]['target_spawn_time'] = None
                self.traffic_flow_info[tf]['vehicle_sensor'] = []

            # delete actors from world
            self.delete_actors(delete_actors)

        print('Traffic flow is reset, following traffic flow will be activated: ')
        print(self.active_tf_direction)

    def set_tm_params(self, tm_params):
        """
        A setter method,
        Set parameters of the traffic flow.

        We use method name from carla PythonAPI as the dict key!
        """
        for key in tm_params:
            if key in self.tm_params.keys():
                # update keys in current instance
                self.tm_params[key] = tm_params[key]

        print('Traffic flow param is reset.')
        print(self.tm_params)

    def check_distance_exceed(self, vehicle, max_dist=75.):
        """
        Check vehicle is too far from junction center.

        params: max_dist: max distance for keeping a vehicle
        """
        junction_center = self.junction.bounding_box.location
        dist = junction_center.distance(vehicle.get_transform().location)
        if dist >= max_dist:
            return True
        else:
            return False

    def check_in_junction(self, actor: carla.Actor, expand=10.0):
        """
        Check if an actor is contained in the target junction

        param actor: actor to be checked if contained in junction
        param expand: expand distance of the junction bounding box
        """
        contain_flag = False

        bbox = self.junction.bounding_box
        actor_loc = actor.get_location()

        relative_vec = actor_loc - bbox.location  # carla.Vector3D

        # todo check if junction bbox has a rotation
        if (relative_vec.x <= bbox.extent.x + expand) and \
                (relative_vec.x >= -bbox.extent.x - expand) and \
                (relative_vec.y >= -bbox.extent.y - expand) and \
                (relative_vec.y <= bbox.extent.y + expand):
            contain_flag = True

        return contain_flag

    def check_blocked(self, vehicle_info: dict, max_block_time=5.0):
        """
        Check if a vehicle is blocked,
        usually blocked in junction, or blocked by other traffic lights.

        todo add args to set maximum block time in and out of the junction

        param vehicle: vehicle to be checked
        param max_block_time: the maximum block time for npc vehicle
                              which is outside the junction

        return: bool, True refers that the vehicle is blocked
        """
        vehicle = vehicle_info['vehicle']
        block_time = vehicle_info['block_time']

        # check if vehicle is blocked
        block_flag = False

        # check if vehicle is in the junction
        in_junction = self.check_in_junction(vehicle)

        if in_junction:
            if block_time >= self.red_phase_time:
                self.visualize_actor(vehicle, color=red, thickness=1.0)
                block_flag = True
        else:
            if block_time >= max_block_time:
                self.visualize_actor(vehicle, color=red, thickness=1.0)
                block_flag = True

        return block_flag

    def check_collision(self, vehicle_info: dict):
        """
        Check if a vehicle collides with other vehicle.
        """
        vehicle = vehicle_info['vehicle']
        sensor_api = vehicle_info['sensor_api']
        collision_flag = sensor_api.collision_flag

        if collision_flag:
            self.visualize_actor(vehicle, color=red, thickness=1.0)

        return collision_flag

    def delete_actors(self, delete_list):
        """
        Delete actors stored in list.
        """
        actor_id = [x.id for x in delete_list]
        # carla.Client.apply_batch_sync method
        response_list = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in delete_list], True)
        failures = []
        for response in response_list:
            if response.has_error():
                failures.append(response)

        if self.debug:
            if not failures:
                print('Following actors are destroyed.')
                print(actor_id)
            else:
                print('Fail to delete: ')
                print(failures)

    def update_block_time(self):
        """
        todo add a method to check traffic flow vehicles delete condition

        Update the block time of all active traffic flow vehicles.

        This method is supposed to be called each timestep.
        """
        # find info dict of this vehicle
        for index, item in enumerate(self.vehicle_info_list):
            vehicle = item[0]
            vehicle_info = item[1]
            # get current velocity
            speed = get_speed(vehicle)  # in km/h
            if speed <= 0.05:  # check by a threshold
                # todo this method need to be checked
                vehicle_info['block_time'] += self.timestep_length
            else:
                # reset the block time
                self.vehicle_info_list[index][1]['block_time'] = 0

    def delete_npc(self):
        """
        Check and delete unused npc vehicle.
        """
        # update block time of all NPC vehicles
        self.update_block_time()

        delete_list = []
        # info_tuple is a tuple of vehicle and vehicle_info dict
        for info_tuple in self.vehicle_info_list:
            vehicle = info_tuple[0]
            info_dict = info_tuple[1]
            sensor = info_dict['sensor']

            # conditions for removing this vehicle, delete if any is True
            distance_cond = self.check_distance_exceed(vehicle)
            block_cond = self.check_blocked(info_dict)
            collision_cond = self.check_collision(info_dict)
            conditions = [distance_cond, block_cond, collision_cond]

            if any(conditions):
                delete_list.append(vehicle)
                delete_list.append(sensor)

                # clear storage info
                self.npc_vehicles.remove(vehicle)
                self.vehicle_info_list.remove(info_tuple)
                # remove vehicle sensor tuple from npc_info dict
                for key, item in self.traffic_flow_info.items():
                    veh_sen_list = item['vehicle_sensor']
                    for sensor_tuple in veh_sen_list:  # tup is (vehicle, sensor)
                        veh = sensor_tuple[0]
                        if veh.id == vehicle.id:
                            veh_sen_list.remove(sensor_tuple)

        if delete_list:
            self.delete_actors(delete_list)

    def get_time(self):
        """
        Get current time using timestamp(carla.Timestamp)
        :return: current timestamp in seconds.
        """
        worldsnapshot = self.world.get_snapshot()
        timestamp = worldsnapshot.timestamp
        now_time = timestamp.elapsed_seconds

        return now_time

    def get_time_interval(self):
        """
        Time interval till next vehicle spawning in seconds.
        :return: time_interval, float
        """
        time_interval = random.uniform(self.lower_limit, self.upper_limit)

        return time_interval

    def spawn_single_vehicle(self, transform, name=None):
        """
        Spawn single NPC vehicle at given transform.

        todo add arg to set vehicle model

        :return: carla.Actor, spawned npc vehicle
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            color = '0, 255, 255'  # use string to identify a RGB color
            bp.set_attribute('color', color)
        if name:
            bp.set_attribute('role_name', name)  # set actor name

        # spawn npc vehicle
        try:
            vehicle = self.world.spawn_actor(bp, transform)  # use spawn method
            # necessary to tick world, seem not necessary to sleep
            self.world.tick()
            # time.sleep(0.05)
        except:
            raise RuntimeError('Fail to spawn a NPC vehicle, please check.')

        # create collision sensor
        sensor = Sensors(self.world, vehicle)

        # store vehicle info into dict
        info_dict = {
            'id': vehicle.id,
            'vehicle': vehicle,  # carla.Vehicle
            'sensor_api': sensor,  # instance of Sensor
            'sensor': sensor.collision,  # carla.Actor
            'block_time': 0.,  # the time this vehicle has been blocked
        }

        return vehicle, info_dict

    def set_collision_detection(self, reference_actor, other_actor):
        """
        Set the collision detection of a npc vehicle with another actor.
        """

        collision_probability = self.tm_params['collision_probability']
        # ignore conflict with ego vehicle with a probability
        if np.random.random() <= collision_probability:
            collision_detection_flag = False  # refers to enable collision
        else:
            collision_detection_flag = True
        # reference_actor: to be set; other_actor: target vehicle
        self.traffic_manager.collision_detection(reference_actor, other_actor, collision_detection_flag)

        collision_statement = '' if collision_detection_flag else 'not'
        # print('Vehicle ', reference_actor.id, ' is ', collision_statement, 'enabled to collide with ego vehicle.')

    def set_traffic_manager(self, vehicle, tm_params: dict):
        """
        Register vehicle to the traffic manager with the given setting.

        todo add api tp set different traffic manager params

        :param vehicle: target vehicle(npc)
        :param tm_params: traffic manager parameters
        """
        # set traffic manager for each vehicle
        vehicle.set_autopilot(True, int(self.tm_port))  # Doc is wrong, the 2nd optional arg is tm_port

        # traffic lights
        per = tm_params['ignore_lights_percentage']
        self.traffic_manager.ignore_lights_percentage(vehicle, per)

        # speed limits
        per = self.get_percentage_by_target_speed(vehicle, tm_params['target_speed'])
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, per)

        # set collision detection for ego
        if self.ego_vehicle:
            self.set_collision_detection(vehicle, self.ego_vehicle)

        # time.sleep(0.1)
        # print('Vehicle ', vehicle.id, ' is set to traffic manager ', self.tm_port)

    def spawn_new_vehicles(self):
        """
        Spawn all traffic flow in this junction.
        3 flows if crossroad, 2 flows if T-road
        """
        for key, item in self.traffic_flow_info.items():
            # only generate active tf
            if key not in self.active_tf_direction:
                continue
            # get spawn waypoint
            transform = item['spawn_transform']
            transform.location.z = 0.5

            # todo add param to tune time interval of different traffic flow
            # set initial spawn time
            if not item['target_spawn_time']:
                item['target_spawn_time'] = 0

            # ========== conditions of spawning vehicle ==========
            # condition of distance to start location
            if item['vehicle_sensor']:
                last_vehicle = item['vehicle_sensor'][-1][0]
                # last_spawned_vehicle =self.straight_npc_list[-1]
                distance = transform.location.distance(last_vehicle.get_transform().location)
                # check if distance gap is large enough
                distance_rule = distance >= self.distance_threshold
                # if distance rule is not satisfied, this direction is skipped
                if not distance_rule:
                    continue
            else:
                distance_rule = True

            # condition of gap time
            now_time = self.get_time()
            if now_time >= item['target_spawn_time']:
                time_rule = True
            else:
                time_rule = False

            # todo add a probability to spawn npc vehicles
            if distance_rule and time_rule:
                try:
                    vehicle, info_dict = self.spawn_single_vehicle(transform)
                    sensor = info_dict['sensor']  # sensor actor

                    # register spawned vehicle to traffic manager
                    self.set_traffic_manager(vehicle, self.tm_params)

                    # append to all storage
                    self.npc_vehicles.append(vehicle)  # add to npc vehicles list
                    self.vehicle_info_list.append((vehicle, info_dict))
                    item['vehicle_sensor'].append((vehicle, sensor))

                    # update last spawn time when new vehicle spawned
                    item['last_spawn_time'] = self.get_time()
                    # min time to spawn next vehicle
                    item['target_spawn_time'] = item['last_spawn_time'] + self.get_time_interval()
                except:
                    print("Fail to spawn a new NPC vehicle and register traffic manager, please check.")
                    if self.debug:
                        raise RuntimeError('Check failure of spawn NPC vehicles...')

    def visualize_actor(self, vehicle, thickness=0.5, color=red, duration_time=0.1):
        """
        Visualize an vehicle(or walker) in server by drawing its bounding box.
        """
        bbox = vehicle.bounding_box
        bbox.location = bbox.location + vehicle.get_location()
        transform = vehicle.get_transform()
        # vehicle bbox fixed to ego coordinate system
        rotation = transform.rotation
        # draw the bbox of vehicle
        self.debug_helper.draw_box(box=bbox,
                                   rotation=rotation,
                                   thickness=thickness,
                                   color=color,
                                   life_time=duration_time)

    # ==================   staticmethod   ==================

    @staticmethod
    def get_junction_by_location(carla_map, location):
        """
        Get junction instance by location.

        todo merge the method to CarlaModule

        param location: coords in carla.Location
        """
        print('The default junction of Town03 will be set.')
        junction_waypoint = carla_map.get_waypoint(location)
        junction = junction_waypoint.get_junction()

        return junction

    @staticmethod
    def get_percentage_by_target_speed(veh, target_speed):
        """
        Calculate vehicle_percentage_speed_difference according to the given speed.

        :param veh: vehicle
        :param target_speed: target speed of the vehicle in km/h
        :return: per: vehicle_percentage_speed_difference, float value refer to a percentage(per %)
        """
        # target speed in m/s
        target_speed = target_speed / 3.6
        speed_limit = veh.get_speed_limit()  # in m/s
        per = (speed_limit - target_speed) / speed_limit

        return per

    @staticmethod
    def junction_contains(location, bbox, margin: float = 5.0):
        """
        Check if a location is contained in a junction bounding box.

        todo: consider a rotated junction

        :param location: location point to check
        :param bbox: junction bounding box(carla.BoundingBox)
        :param margin: detection is larger than the original bbox with margin distance

        :return: bool
        """
        contain_flag = False

        if margin < 0:
            print('margin value must larger than 0.')
        margin = np.clip(margin, 0, float('inf'))

        # relative location to junction center
        _location = location - bbox.location
        extent = bbox.extent

        if extent.x + margin >= _location.x >= -extent.x - margin and \
                extent.y + margin >= _location.y >= -extent.y - margin:
            contain_flag = True

        return contain_flag

    # ==================   In developing   ==================

    def get_npc_spawn_point(self):
        """
        Get npc vehicle spawn point of a junction scenario.
        Current method is designed based on Town03 junction.

        todo fix this method
        todolist:

         - use carla.Junction as input arg
         - get spawn location on the lane with maximum distance to the junction bbox
         - return a list of carla.Waypoint or carla.Transform

        """

        # plot junction coord frame
        location = junction_center_location
        rotation = carla.Rotation()  # default rotation is world coordinate frame
        transform = carla.Transform(location, rotation)
        plot_coordinate_frame(self.world, transform)  # plot coord frame
        # self.set_spectator(carla.Transform(location, start_rotation))

        # using target waypoint of ego vehicle to get spawn location
        start_waypoint = self.map.get_waypoint(start_location)  # start waypoint of this junction
        left_target_waypoint = generate_target_waypoint(start_waypoint, -1)
        right_target_waypoint = generate_target_waypoint(start_waypoint, 1)
        straight_target_waypoint = generate_target_waypoint(start_waypoint, 0)

        draw_waypoint(self.world, left_target_waypoint)
        draw_waypoint(self.world, right_target_waypoint)
        draw_waypoint(self.world, straight_target_waypoint)

        lane_width = left_target_waypoint.lane_width  # consider each lane width is same, usually 3.5

        # ==================================================
        # get npc vehicle spawn location
        # todo add this into class attribute
        lon_dist = 30  # distance to the junction

        # left side
        location_left = left_target_waypoint.transform.location
        d_x = +1 * left_target_waypoint.lane_width * 3
        d_y = +1 * lon_dist
        location_left_2 = location_left + carla.Location(x=d_x, y=d_y)

        left_start_waypoint = self.map.get_waypoint(location_left_2)
        draw_waypoint(self.world, left_start_waypoint, color=[red, green])

        # right side
        location_right = right_target_waypoint.transform.location
        d_x = -1 * right_target_waypoint.lane_width * 3
        d_y = -1 * lon_dist
        location_right_2 = location_right + carla.Location(x=d_x, y=d_y)

        right_start_waypoint = self.map.get_waypoint(location_right_2)
        draw_waypoint(self.world, right_start_waypoint, color=[red, green])

        # straight side
        location_straight = straight_target_waypoint.transform.location
        d_y = +1 * straight_target_waypoint.lane_width
        d_x = -1 * lon_dist
        location_straight_2 = location_straight + carla.Location(x=d_x, y=d_y)

        straight_start_waypoint = self.map.get_waypoint(location_straight_2)
        draw_waypoint(self.world, straight_start_waypoint, color=[red, green])

        # todo: pack result
        # npc_spawn_transform_dict

        print("npc spawn location is generated.")
