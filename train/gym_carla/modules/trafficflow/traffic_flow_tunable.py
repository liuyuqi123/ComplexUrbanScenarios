"""
Inherit from trafficflow_3. Major modifications:

 - Add attributes for tuning the complexity of traffic flow
 -

"""

from gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

# ==================================================
# import carla module
import glob
import os
import sys

carla_root = os.path.join(root_path, 'CARLA_'+carla_version)
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

import time
import numpy as np
import random

from gym_carla.util_development.scenario_helper_modified import generate_target_waypoint
from gym_carla.util_development.util_junction import (plot_coordinate_frame)
from gym_carla.util_development.util_visualization import draw_waypoint
from gym_carla.navigation.misc import get_speed

# import collision sensors
from gym_carla.util_development.sensors import Sensors

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)

start_location = carla.Location(x=53.0, y=128.0, z=3.0)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)

# default Town03 junction center location
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)


class TrafficFlowTunable:
    """
    This class is responsible for generation and management of a traffic flow around a junction.
    """

    # time interval to spawn new vehicles
    lower_limit = 1.0
    upper_limit = 10.0

    # todo use probability to spawn npc vehicles, ref on sumo method
    prob_dict = {
        'left': 0.11,
        'mid': 0.11,
        'right': 0.11,
    }

    # ==================================================
    # these info is for the junction in Town 03
    # dict to record npc vehicle info

    # todo this dict is supposed to keep all relative info about traffic flow, should add a api for param transfer

    # todo get the traffic flow spawn point by a script
    # use carla coordinate frame axis
    npc_info = {
        'positive_x': {
            'transform': carla.Transform(carla.Location(x=53.0, y=128.0, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],
        },

        'positive_y': {
            'transform': carla.Transform(carla.Location(x=5.767616, y=175.509048, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=269.637451, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],  # npc vehicles which in the junction for state representation
        },

        'negative_y': {
            'transform': carla.Transform(carla.Location(x=-6.268355, y=90.840492, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=89.637459, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],
        },

        'negative_x': {
            'transform': carla.Transform(carla.Location(x=-46.925552, y=135.031494, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=-1.296802, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],
        }

    }

    # ==================================================
    # abandoned
    # # use namedtuple to store vehicle related api
    # VehInfo = namedtuple('VehInfo', ['id', 'actor', 'sensor', 'sensor_api', 'block_time'])
    # ==================================================

    # # element of a vehicle info dict, key is the actor id of vehicle
    # veh_info_elem = {
    #     'id': None,
    #     'actor': None,
    #     'sensor_api': None,
    #     'sensor': None,
    #     'block_time': None,
    # }

    def __init__(self, carla_api, junction=None, route_option='left', start_direction='positive_y'):
        """
        todo fix the api to determine the start lane of ego vehicle
        should coordinate with the settings in env, set to 4 options:
        positive / negative - x / y (using carla coordinate frame)

        start_direction should combine value(relative location) and str

        """
        # get API
        self.carla_api = carla_api
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        settings = self.world.get_settings()

        # time step length of carla simulator
        self.timestep_length = settings.fixed_delta_seconds

        # route task
        self.route_option = route_option
        self.start_direction = start_direction

        # assign junction
        if junction:
            self.junction = junction
        else:
            self.junction = self.get_junction_by_coord(junction_center)

        # ego info
        self.ego_vehicle = None  # ego vehicle in env
        self.ego_id = None

        # all npc vehicles spawned by traffic flow
        self.npc_vehicles = []  # using list to manage npc vehicle
        self.count = 0  # total number of alive npc vehicles

        # init the dict to store npc info
        # remove the direction of ego vehicle
        self.init_info_dict(start_direction)  # this line will updated npc info dict

        # collision sensors for each vehicle in traffic flow
        # todo use namedtuple to manage vehicle and its collision sensor?
        self.sensors = []

        # list to store vehicle and its collision sensor, element: (veh, sensor)
        self.veh_tuple_list = []

        # store information of all vehicles
        # element: dict stores information of a single vehicle
        # self.veh_info_elem
        self.vehicle_info_list = []

    def init_info_dict(self, start_direction):
        """
        Init info dict by given args.

        Traffic flow will be generated in the rest 3 directions
        """
        # must re-assign key and value in a iteration loop
        self.npc_info = {k: v for k, v in self.npc_info.items() if k is not start_direction}
        print('npc info dict is updated.')

    def set_info_dict(self, tf_direction: list = None):
        """
        Set the traffic flow with a dict

        params: tf_direction: selected traffic flow direction

        """
        self.npc_info = {k: v for k, v in self.npc_info.items() if k in tf_direction}

        # self.npc_info = {k: v for k, v in self.npc_info.items() if k is not start_direction}
        # print('npc info dict is updated.')

        print('traffic flow is reset.')

    def get_junction_by_coord(self, location):
        """
        Get junction by coordinate location.
        If the location given is in a certain junction.

        param location: coords in carla.Location
        """
        junction_waypoint = self.map.get_waypoint(location)
        junction = junction_waypoint.get_junction()

        return junction

    def retrieve_ego_vehicle(self):
        """
        Retrieve ego vehicle from world.
        """
        self.npc_vehicles = []
        self.ego_vehicle = None

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a actorlist instance, iterable

        if vehicle_list:
            for veh in vehicle_list:

                attr = veh.attributes  # dict
                # ego vehicle is excluded
                if veh.attributes['role_name'] == 'ego':
                    self.ego_vehicle = veh
                    self.ego_id = veh.id
                else:
                    self.npc_vehicles.append(veh)
            print('Vehicles are updated.')
            if not self.ego_vehicle:
                print('ego vehicle not found.')

    def draw_waypoint(self, waypoint, color=(red, green)):
        """
        An internal interface
        """
        draw_waypoint(self.world, waypoint, color=color)

    def get_npc_spawn_point(self):
        """
        Get npc vehicle spawn point in a junction scenario.
        Current method is designed based on Town03 junction.

        todo: fix this method
        todolist:

         - use carla.Junction as input arg
         - get spawn location on the lane with maximum distance to the junction bbox
         - return a list of carla.Waypoint or carla.Transform


        """

        # plot junction coord frame
        location = junction_center
        rotation = carla.Rotation()  # default rotation is world coordinate frame
        transform = carla.Transform(location, rotation)
        plot_coordinate_frame(self.world, transform)  # plot coord frame
        # self.set_spectator(carla.Transform(location, start_rotation))

        # using target waypoint of ego vehicle to get spawn location
        start_waypoint = self.map.get_waypoint(start_location)  # start waypoint of this junction
        left_target_waypoint = generate_target_waypoint(start_waypoint, -1)
        right_target_waypoint = generate_target_waypoint(start_waypoint, 1)
        straight_target_waypoint = generate_target_waypoint(start_waypoint, 0)

        self.draw_waypoint(left_target_waypoint)
        self.draw_waypoint(right_target_waypoint)
        self.draw_waypoint(straight_target_waypoint)

        lane_width = left_target_waypoint.lane_width  # consider each lane width is same, usually 3.5

        # ==================================================
        # get npc vehicle spawn location

        lon_dist = 30  # distance to the junction

        # left side
        location_left = left_target_waypoint.transform.location
        d_x = +1 * left_target_waypoint.lane_width * 3
        d_y = +1 * lon_dist
        location_left_2 = location_left + carla.Location(x=d_x, y=d_y)

        left_start_waypoint = self.map.get_waypoint(location_left_2)
        self.draw_waypoint(left_start_waypoint, color=[red, green])

        # right side
        location_right = right_target_waypoint.transform.location
        d_x = -1 * right_target_waypoint.lane_width * 3
        d_y = -1 * lon_dist
        location_right_2 = location_right + carla.Location(x=d_x, y=d_y)

        right_start_waypoint = self.map.get_waypoint(location_right_2)
        self.draw_waypoint(right_start_waypoint, color=[red, green])

        # straight side
        location_straight = straight_target_waypoint.transform.location
        d_y = +1 * straight_target_waypoint.lane_width
        d_x = -1 * lon_dist
        location_straight_2 = location_straight + carla.Location(x=d_x, y=d_y)

        straight_start_waypoint = self.map.get_waypoint(location_straight_2)
        self.draw_waypoint(straight_start_waypoint, color=[red, green])

        # todo: pack result
        # npc_spawn_transform_dict

        print("npc spawn location is generated.")

    def set_autopilot(self, vehicle, p_collision):
        """
        Set autopilot for specified vehicle

        todo add API to tune the behavior

        :param vehicle: target vehicle(npc)
        :param p_collision: probability to avoid collision with ego vehicle
        """
        # set traffic manager for each vehicle
        tm_port = self.traffic_manager.get_port()  # check tm_port number
        vehicle.set_autopilot(True, int(tm_port))  # Doc is wrong, optional arg is tm_port
        # ignore traffic lights
        self.traffic_manager.ignore_lights_percentage(vehicle, 100.0)
        # test to show speed limit
        speed_limit = vehicle.get_speed_limit()
        # set speed (limit)
        percentage = -1.0
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, percentage)
        # collision detection with ego vehicle
        if self.ego_vehicle:
            # ignore conflict with ego vehicle with a probability
            if np.random.random() <= p_collision:
                collision_detection_flag = False
                print("This vehicle will NOT avoid collision with ego vehicle.")
            else:
                collision_detection_flag = True
                print("This vehicle will avoid collision with ego vehicle.")
            self.traffic_manager.collision_detection(vehicle, self.ego_vehicle, collision_detection_flag)

        time.sleep(0.1)
        # print('traffic manager is set.')

    def spawn_npc(self, transform, name=None):
        """
        Spawn a npc vehicle at given transform
        :return: carla.Actor, spawned npc vehicle
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            color = '0, 255, 255'  # use string to identify a RGB color
            bp.set_attribute('color', color)

        if name:
            bp.set_attribute('role_name', name)  # set actor name

        # todo check usage of sticky control in sync mode
        """
        when “sticky_control” is “False”, 
        the control will be reset every frame to 
        its default (throttle=0, steer=0, brake=0) values.
        """
        # bp.set_attribute('sticky_control', 'False')

        # spawn npc vehicle
        vehicle = self.world.spawn_actor(bp, transform)  # use spawn method
        self.world.tick()
        time.sleep(0.1)
        if vehicle:
            # print("Number", vehicle.id, "npc vehicle is spawned.")  # actor id number of this vehicle
            pass
        else:
            print('Fail to spawn vehicle.')
            raise

        # print('')

        # create collision sensor
        sensor = Sensors(self.world, vehicle)
        self.sensors.append(sensor.collision)  # just sensor actor
        self.veh_tuple_list.append((vehicle, sensor))  # Sensor instance

        # store vehicle info into dict
        info_dict = {
            'id': vehicle.id,
            'vehicle': vehicle,
            'sensor_api': sensor,
            'sensor': sensor.collision,
            'block_time': 0.,
        }
        self.vehicle_info_list.append(info_dict)

        return vehicle

    def get_time_interval(self):
        """
        Time interval till next vehicle spawning in seconds.
        :return: time_interval, float
        """
        time_interval = random.uniform(self.lower_limit, self.upper_limit)

        return time_interval

    def get_time(self):
        """
        Get current time using timestamp(carla.Timestamp)
        :return: current timestamp in seconds.
        """
        worldsnapshot = self.world.get_snapshot()
        timestamp = worldsnapshot.timestamp
        now_time = timestamp.elapsed_seconds

        return now_time

    def distance_exceed(self, vehicle):
        """
        Check if distance to junction origin exceed limits.

        todo: junction center as class attribute
        """
        limit = 65.0
        dist = junction_center.distance(vehicle.get_transform().location)
        if dist >= limit:
            return True
        else:
            return False

    def update_info(self):
        """
        Update info of all active vehicles.

        This method is supposed to be called each timestep.

        Function:
         - Update the block time of each vehicle
         - todo others
        """
        # find info dict of this vehicle
        for item in self.vehicle_info_list:
            vehicle = item['vehicle']
            # get current velocity
            speed = get_speed(vehicle)  # in km/h
            if speed <= 0.05:  # check by a threshold
                # todo this method need to be checked
                item['block_time'] += self.timestep_length
            else:
                # reset the block time
                item['block_time'] = 0

    def block_check(self, vehicle: carla.Vehicle, max_block_time=10.0):
        """
        Check if a vehicle is blocked(in a junction).

        param vehicle: vehicle to be checked
        param max_block_time: the maximum time a vehicle supposed to be blocked

        return: bool, True if the vehicle is blocked
        """
        block_flag = False

        vehicle_info_dict = None
        for item in self.vehicle_info_list:
            if vehicle.id == item['id']:
                vehicle_info_dict = item
                break

        # todo assume that there is one vehicle at least
        if not vehicle_info_dict:
            print('Fail to check if vehicle blocked.')
            return False

        # todo if a vehicle is blocked outside a junctions
        # # check if vehicle is in a junction
        # veh_loc = vehicle.get_location()
        # current_wp = self.map.get_waypoint(location=veh_loc,
        #                                    project_to_road=True,  # must set to True to avoid error(None)
        #                                    lane_type=carla.LaneType.Driving)
        # in_junction = current_wp.is_junction  # class attribute of waypoint

        in_junction = self.check_in_junction(vehicle)

        block_time = vehicle_info_dict['block_time']
        # todo fix the conditions for block
        if in_junction and (block_time >= max_block_time):
            self.visualize_actor(vehicle, color=magenta, thickness=1.0)
            block_flag = True

        return block_flag

    def collision_check(self, vehicle):
        """
        Check if a vehicle collides.
        """
        sensor = None
        for tup in self.veh_tuple_list:
            if vehicle.id == tup[0].id:
                sensor = tup[1]
                # print('collision sensor found')

        collision_flag = sensor.collision_flag

        # debug
        if collision_flag:
            self.visualize_actor(vehicle, color=red, thickness=1.0)
            # print('')

        return collision_flag

    def delete_npc(self):
        """
        Check and delete unused npc vehicle.
        """
        delete_list = []

        # check if beyond max distance
        for vehicle in self.npc_vehicles:
            # todo merge the conditions with an api
            if self.distance_exceed(vehicle) or self.collision_check(vehicle) or self.block_check(vehicle):
                delete_list.append(vehicle)
                self.npc_vehicles.remove(vehicle)

                # remove vehicle from npc_info dict
                for key in self.npc_info:
                    for actor in self.npc_info[key]['actor_list']:
                        if vehicle.id == actor.id:
                            self.npc_info[key]['actor_list'].remove(actor)

        # delete collision sensors as well
        for veh in delete_list:
            # sync veh_tuple_list
            for tup in self.veh_tuple_list:
                if veh == tup[0]:
                    sensor = tup[1].collision
                    delete_list.append(sensor)
                    self.veh_tuple_list.remove(tup)

            # sync vehicle_info_list
            for elem in self.vehicle_info_list:
                if veh.id == elem['id']:
                    self.vehicle_info_list.remove(elem)

        # print('NPC vehicles checked. Deleting vehicles...')

        if delete_list:
            # self.client.apply_batch([carla.command.DestroyActor(x) for x in delete_list])
            # print(self.client.apply_batch([carla.command.DestroyActor(x) for x in delete_list]))

            # carla.Client.apply_batch_sync method will not crash
            response_list = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in delete_list], True)

            # attribute_error = response_list[0].error  # attribute error is not working
            method_error = response_list[0].has_error()
            if not method_error:
                print('npc vehicle', delete_list[0].id, "is destroyed")

            # for x in delete_list:
            #     carla.command.DestroyActor(x)

            # check if actor is destroyed successfully
            # actor_list = self.world.get_actors()
            # vehicle_list = self.world.get_actors().filter('vehicle.*')

        # print("vehicle", vehicle.id, "is destroyed")
        # delete_list.clear()

        # print('d')

        # ==================================================
        # original method
        # todo: memory leak for unknown reason
        """
        delete_list = []
        for vehicle in self.npc_vehicles:
            if self.distance_exceed(vehicle):
                delete_list.append(vehicle.id)
                self.npc_vehicles.remove(vehicle)

        if delete_list:
            # delete vehicle actor
            self.client.apply_batch([carla.command.DestroyActor(x) for x in delete_list])
            for x in delete_list:
                print("vehicle", x, "is deleted")

            # remove vehicle from all npc list
            for actor_id in delete_list:
                # npc_vehicles
                for actor in self.npc_vehicles:
                    if actor_id == actor.id:
                        self.npc_vehicles.remove(actor)

                for key in self.npc_info:
                    for actor in self.npc_info[key]['actor_list']:
                        if actor_id == actor.id:
                            self.npc_info[key]['actor_list'].remove(actor)
                print("vehicle", actor_id, "is deleted")

        delete_list.clear()        
        """
        # ==================================================

        # print('delete method runned.')

    def spawn_all_traffic_flow(self):
        """
        Spawn all traffic flow in this junction.
        3 tf if crossroad, 2 tf if T-road

        todo use a probability to spawn npc vehicles
        """
        for key in self.npc_info:

            transform = self.npc_info[key]['transform']
            transform.location.z = 1.0

            # set initial spawn time
            if not self.npc_info[key]['target_spawn_time']:
                self.npc_info[key]['target_spawn_time'] = 0

            # todo: add API to get a specified time interval for different flow
            # lower_limit = 1.0
            # upper_limit = 5.0
            # time_interval = random.uniform(lower_limit, upper_limit)
            # self.target_spawn_time = self.last_spawn_time + time_interval

            # check spawn condition of spawning vehicle
            # distance rule
            if self.npc_info[key]['actor_list']:
                # last_spawned_vehicle =self.straight_npc_list[-1]
                distance = transform.location.distance(self.npc_info[key]['actor_list'][-1].get_transform().location)
                distance_threshold = 5.0
                distance_rule = distance >= distance_threshold
                if distance_rule:
                    pass
            else:
                distance_rule = True

            # check if time rule satisfied
            now_time = self.get_time()
            if now_time >= self.npc_info[key]['target_spawn_time']:
                time_rule = True
            else:
                time_rule = False

            if distance_rule and time_rule:
                name = key + str(self.npc_info[key]['count'] + 1)  # name of the actor to be spawned
                try:
                    vehicle = self.spawn_npc(transform, name)
                    # set autopilot for ego vehicle
                    p_collision = 0.5  # probability of NOT avoiding collision with ego vehicle

                    # todo check if set autopilot
                    self.set_autopilot(vehicle, p_collision)

                    self.npc_vehicles.append(vehicle)  # add to npc vehicles list
                    self.npc_info[key]['actor_list'].append(vehicle)  # only store id
                    self.npc_info[key]['count'] += 1
                    self.count += 1
                    # update last spawn time when new vehicle spawned
                    self.npc_info[key]['last_spawn_time'] = self.get_time()
                    # time interval till spawn next vehicle
                    self.npc_info[key]['target_spawn_time'] = self.npc_info[key]['last_spawn_time'] \
                                                              + self.get_time_interval()
                    # return vehicle
                except:
                    print("fail to spawn a npc vehicle, please check.")
                    # return None

            # print('testing spawn traffic flow')

        # print('spawn new npc checked.')

    def visualize_actor(self, vehicle, thickness=0.5, color=red, duration_time=0.1):
        """
        Visualize an vehicle(or walker) in server by drawing its bounding box.
        """
        bbox = vehicle.bounding_box
        bbox.location = bbox.location + vehicle.get_location()
        transform = vehicle.get_transform()
        # todo usually vehicle bbox has not rotation, check it in future
        rotation = transform.rotation

        self.debug_helper.draw_box(box=bbox,
                                   rotation=rotation,
                                   thickness=thickness,
                                   color=color,
                                   life_time=duration_time)

        # for debug
        # self.world.tick()

    def check_in_junction(self, actor: carla.Actor, expand=3.0):
        """
        Check if an actor is contained in the target junction

        todo generalization for other junction scenario

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
            # for debug
            # self.visualize_actor(actor, color=yellow)
            contain_flag = True

        return contain_flag

    @staticmethod
    def junction_contains(location, bbox):
        """
        Check if a location is contained in a junction bounding box.

        todo: consider a rotated junction, with a yaw angle

        :param location: location point to check
        :param bbox: junction bounding box(carla.BoundingBox)
        :return: bool
        """

        vector = location - bbox.location  # location relative to junction center
        extent = bbox.extent

        if extent.x >= vector.x >= -extent.x and \
                extent.y >= vector.y >= -extent.y:
            return True
        else:
            return False

    @staticmethod
    def junction_area_contains(location, bbox):
        """
        Check if a location(vehicle object) is contained in a junction area.

        todo 1. consider a rotated junction, with a yaw angle
        todo 2. plot area box

        :param location: location point to check
        :param bbox: junction bounding box(carla.BoundingBox)
        :return: bool
        """
        expanded_distance = 10.0

        vector = location - bbox.location  # location relative to junction center
        extent = bbox.extent

        if extent.x+expanded_distance >= vector.x >= -extent.x-expanded_distance and \
                extent.y+expanded_distance >= vector.y >= -extent.y-expanded_distance:
            return True
        else:
            return False

    def get_near_npc(self):
        """
        Get npc vehicles which is in junction.
        actor list will be updated each time method is called.
        """
        near_npc_dict = {
            'left': [],
            'right': [],
            'straight': [],
        }

        for key in self.npc_info:
            for actor in self.npc_info[key]['actor_list']:
                # check if vehicle is in the junction
                if self.junction_contains(actor.get_location(), self.junction.bounding_box):
                    # self.npc_info[key]['nearby_npc'].append(actor)
                    near_npc_dict[key].append(actor)

        return near_npc_dict

    def get_near_npc2(self):
        """
        Get 2 npc vehicles of each traffic flow in expanded junction area

        todo merge this 2 methods.
        :return:
        """
        near_npc_dict = {
            'left': [],
            'right': [],
            'straight': [],
        }

        for key in self.npc_info:
            for num, actor in enumerate(self.npc_info[key]['actor_list']):
                # check if vehicle is in the expanded junction area
                if self.junction_area_contains(actor.get_location(), self.junction.bounding_box):
                    # self.npc_info[key]['nearby_npc'].append(actor)
                    near_npc_dict[key].extend(self.npc_info[key]['actor_list'][num:])
                    break

            # if len(near_npc_dict[key]) < 2:
            #     # need to check when calling
            #     if self.npc_info[key]['actor_list'][num+1]:
            #         near_npc_dict[key].append(self.npc_info[key]['actor_list'][num+1])

        return near_npc_dict

    def get_npc_dict(self):
        """
        Get npc vehicles of all 3 direction.
        :return: npc dict
        """
        npc_dict = {
            'left': [],
            'right': [],
            'straight': [],
        }

        for key in self.npc_info:
            for actor in self.npc_info[key]['actor_list']:
                npc_dict[key].append(actor)

        return npc_dict

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
        # task 1: check if new vehicles should be spawned
        self.spawn_all_traffic_flow()

        # update vehicle info
        self.update_info()

        # task 2: check and delete npc vehicles
        self.delete_npc()

        # todo call retrieve ego vehicle each timestep
        # make npc vehicles be aware of ego vehicle









