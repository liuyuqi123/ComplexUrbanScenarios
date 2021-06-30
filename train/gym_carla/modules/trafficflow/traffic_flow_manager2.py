"""
A developing version of traffic flow manager,
add a few new functions:

 - get spawn point by route generator
 - longer route from -y direction
 - multiple spawn point for traffic flow

"""

import carla

import numpy as np
import random

from gym_carla.util_development.kinetics import get_transform_matrix

from gym_carla.modules.trafficflow.traffic_flow_manager import TrafficFlowManager


class TrafficFlowManager2(TrafficFlowManager):
    """
    A improved version of traffic manager.

    todo improve active direction option
    """

    # time interval to spawn new vehicles
    # tuple: (mean, half_interval)
    time_interval = (3., 1.)

    # min distance to last spawned vehicle
    distance_threshold = 3.0

    # params for traffic manager
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
    tf_directions = ['positive_x', 'negative_x', 'positive_y', 'negative_y_0', 'negative_y_1']

    # todo the params for Town 03 junction requires manually tuning, fix method for other random junction
    # this dict stores all traffic flow information
    # max spawn distance: {negative_x: 45, positive_y: 40(ego direction)}
    traffic_flow_info = {
        'positive_x': {
            'spawn_transform': carla.Transform(carla.Location(x=71.354889, y=130.074112, z=0.018447),
                                               carla.Rotation(pitch=359.836853, yaw=179.182800, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],  # list of tuple, (vehicle, sensor)
        },

        'negative_x': {
            'spawn_transform': carla.Transform(carla.Location(x=-64.222389, y=135.423065, z=0.000000),
                                               carla.Rotation(pitch=0.000000, yaw=-361.296783, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'positive_y': {
            'spawn_transform': carla.Transform(carla.Location(x=2.354240, y=189.210159, z=0.000000),
                                               carla.Rotation(pitch=0.000000, yaw=-90.362534, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'negative_y_0': {
            'spawn_transform': carla.Transform(carla.Location(x=-6.411462, y=68.223877, z=0.000000),
                                               carla.Rotation(pitch=0.000000, yaw=89.637459, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'negative_y_1': {
            'spawn_transform': carla.Transform(carla.Location(x=-9.911532, y=68.223877, z=0.000000),
                                               carla.Rotation(pitch=0.000000, yaw=89.637459, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },
    }

    def __init__(self,
                 carla_api,
                 junction=None,
                 active_tf_direction=None,
                 tm_seed=int(0),
                 debug=False,
                 ):

        super(TrafficFlowManager2, self).__init__(carla_api,
                                                  junction=junction,
                                                  active_tf_direction=active_tf_direction,
                                                  tm_seed=tm_seed,
                                                  debug=debug,
                                                  )

    def set_spawn_point(self, distance):
        """
        todo get spawn point for random intersection
        Get traffic flow spawn point.
        """
        pass

    def clean_up(self):
        """
        Clear all NPC vehicles and their sensors
        without changing other settings.
        """
        delete_actors = []  # actors to delete

        # clean all storage
        self.vehicle_info_list = []
        self.npc_vehicles = []

        # reset the tf dict in npc_info
        for tf in self.traffic_flow_info:

            _vehicle_sensor_list = self.traffic_flow_info[tf]['vehicle_sensor']
            for item in _vehicle_sensor_list:
                vehicle = item[0]
                sensor = item[1]
                delete_actors.append(vehicle)  # vehicle
                delete_actors.append(sensor)  # collision sensor actor

            self.traffic_flow_info[tf]['last_spawn_time'] = 0
            self.traffic_flow_info[tf]['target_spawn_time'] = None
            self.traffic_flow_info[tf]['vehicle_sensor'] = []

        self.delete_actors(delete_actors)

    def get_time_interval(self):
        """
        Time interval till next vehicle spawning in seconds.
        :return: time_interval, float
        """
        lower_limit = self.time_interval[0] - self.time_interval[1]
        upper_limit = self.time_interval[0] + self.time_interval[1]
        time_interval = random.uniform(lower_limit, upper_limit)

        return time_interval

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
        mean_speed = tm_params['target_speed'][0]
        speed_interval = tm_params['target_speed'][1]
        target_speed = random.uniform(mean_speed - speed_interval, mean_speed + speed_interval)
        per = self.get_percentage_by_target_speed(vehicle, target_speed)
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, per)

        # auto lane change
        auto_lane_change = self.tm_params['auto_lane_change']
        self.traffic_manager.auto_lane_change(vehicle, auto_lane_change)

        # minimum distance to leading vehicle
        distance = self.tm_params['distance_to_leading_vehicle']
        self.traffic_manager.distance_to_leading_vehicle(vehicle, distance)

        # set collision detection for ego
        if self.ego_vehicle:
            self.set_collision_detection(vehicle, self.ego_vehicle)

        # set a initial velocity
        self.set_velocity(vehicle, target_speed=target_speed / 3.6 * 0.75)
        # time.sleep(0.1)
        # print('Vehicle ', vehicle.id, ' is set to traffic manager ', self.tm_port)

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
        #     self.world.tick()

    def spawn_new_vehicles(self):
        """
        Spawn all traffic flow in this junction.
        3 flows if crossroad, 2 flows if T-road
        """
        for key, item in self.traffic_flow_info.items():
            # only generate active tf
            if key not in self.active_tf_direction:
                continue

            transform = item['spawn_transform']
            # set spawn height, original height is retrieve from waypoint
            transform.location.z = 0.5
            # transform.location.z += 0.5  # sometimes fails

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
