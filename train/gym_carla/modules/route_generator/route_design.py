"""
Design route in Town03.
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

# original version
from train.gym_carla.envs.BasicEnv import BasicEnv
from train.gym_carla.util_development.util_visualization import draw_waypoint
from train.gym_carla.util_development.scenario_helper_modified import (generate_target_waypoint,
                                                                 get_waypoint_in_distance
                                                                 )
from train.gym_carla.util_development.route_manipulation import interpolate_trajectory

# carla colors
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)

# parameters for scenario
start_location = carla.Location(x=53.0, y=128.0, z=1.5)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)

target_location = carla.Location(x=5.24, y=92.28, z=0.5)

#
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)


class RouteGenerator(BasicEnv):
    """
    This class provides methods to generate routes for junction turning scenario.
    """

    # distance before and after the junction
    route_distance = (10, 10)

    def __init__(self):
        super(RouteGenerator, self).__init__()

        self.junction = None  # carla.Junction
        self.debug = True  # debug flag

    def get_junction_by_location(self, center_location):
        """
        Get a junction instance by a location contained in the junction.

        For the junction of which center coordinate is known.

        param center_location: carla.Location of the junction center
        """
        wp = self.map.get_waypoint(location=center_location,
                                   project_to_road=False,  # not in the center of lane(road)
                                   lane_type=carla.LaneType.Driving)

        self.junction = wp.get_junction()

        # junction center coord

    # todo
    """
    def get_junction_by_start_location(self):
        """"""
        self.junction = self.get_front_junction(self.start_waypoint)

    
    """

    def get_start_waypoints(self, junction):
        """
        Get available start waypoints of a junction.

        Assuming a 4-direction junction.
        """

        bbox = junction.bounding_box
        location = bbox.location
        extent = bbox.extent
        # todo if rotation on junction
        rotation = bbox.rotation  # original rotation of the b box

        # plot the junction
        if self.debug:
            # transform of the junction center
            transform = carla.Transform(location, rotation)
            # todo use plot coord frame to plot
            self.draw_waypoint(transform)
            # bounding box
            self.debug_helper.draw_box(box=bbox,
                                       rotation=rotation,
                                       thickness=0.5,
                                       color=red,
                                       life_time=-1.0)

        lane_width = self.map.get_waypoint(location).lane_width

        # start location
        # sequence as [+x -x +y -y] according to local rotation
        x_shift = -1. * lane_width
        x_slack = 0.75  # slack to fit different junction
        x_start = np.array([
            location.x + (extent.x + self.route_distance[0]),
            location.x - (extent.x + self.route_distance[0]),
            location.x + x_slack * lane_width + x_shift,
            location.x - x_slack * lane_width + x_shift,
        ])

        y_shift = 0. * lane_width
        y_slack = 0.75
        y_start = np.array([
            location.y - y_slack * lane_width + y_shift,
            location.y + y_slack * lane_width + y_shift,
            location.y + (extent.y + self.route_distance[0]),
            location.y - (extent.y + self.route_distance[0]),
        ])

        # plot the fixed coord center, for locate correct lane
        loc = carla.Location(x=location.x + x_shift,
                             y=location.y
                             )

        trans = carla.Transform(loc, rotation)
        self.draw_waypoint(trans, color=(white, white))

        start_wapoints = []
        for i in range(4):
            start_location = carla.Location(x=x_start[i], y=y_start[i], z=0)
            start_waypoint = self.map.get_waypoint(location=start_location,
                                                   project_to_road=True,  # not in the center of lane(road)
                                                   lane_type=carla.LaneType.Driving)

            lane_id = start_waypoint.lane_id

            start_wapoints.append(start_waypoint)
            self.draw_waypoint(start_waypoint)
            print('start waypoint is plot. coord: [',
                  start_waypoint.transform.location.x,
                  start_waypoint.transform.location.y,
                  start_waypoint.transform.location.z,
                  ']'
                  )

        return start_wapoints

    def generate_route(self, start_waypoint, turn_flag=-1):
        """
        Generate a route by given start waypoint.

        param turn_flag: left -1, straight 0, right 1
        """
        # 1. get key waypoint, exit point of the junction
        # get exit waypoint of the junction, with a turn flag
        exit_waypoint = generate_target_waypoint(waypoint=start_waypoint, turn=turn_flag)
        self.draw_waypoint(exit_waypoint, color=(blue, yellow))

        end_waypoint, _ = get_waypoint_in_distance(exit_waypoint, self.route_distance[1])
        self.draw_waypoint(end_waypoint, color=(magenta, yan))

        # additional method to return the complete list of the route in junction
        # return format: (list([waypoint, RoadOption]...), destination_waypoint)
        # wp_list, target_waypoint = generate_target_waypoint_list(waypoint=start_wapoints[2], turn=-1)

        # 2. generate a route by the keypoint
        # this method requires list of carla.Location format
        waypoint_list = [start_waypoint.transform.location,
                         exit_waypoint.transform.location,
                         end_waypoint.transform.location,
                         ]
        gps_route, route = interpolate_trajectory(world=self.world,
                                                  waypoints_trajectory=waypoint_list,
                                                  hop_resolution=1.0)

        # plot generated route
        for i, item in enumerate(route):
            trans = item[0]
            self.draw_waypoint(trans)

        return route


class RouteGenerator_2:
    """
    This is a flexible version.
    """

    route_distance = (10, 10)

    def __init__(self, carla_api):
        self.carla_api = carla_api
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

    def get_junction_by_location(self, center_location):
        """
        Get a junction instance by a location contained in the junction.

        For the junction of which center coordinate is known.

        param center_location: carla.Location of the junction center
        """
        wp = self.map.get_waypoint(location=center_location,
                                   project_to_road=False,  # not in the center of lane(road)
                                   lane_type=carla.LaneType.Driving)
        junction = wp.get_junction()

        return junction

    def get_start_waypoints(self, junction):
        """
        Get available start waypoints of a junction.

        Assuming a 4-direction junction.
        """

        bbox = junction.bounding_box
        location = bbox.location
        extent = bbox.extent
        # todo if rotation on junction
        rotation = bbox.rotation  # original rotation of the b box

        # plot the junction
        # transform of the junction center
        transform = carla.Transform(location, rotation)
        # todo use plot coord frame to plot
        draw_waypoint(self.world, transform)
        # bounding box
        self.debug_helper.draw_box(box=bbox,
                                   rotation=rotation,
                                   thickness=0.5,
                                   color=red,
                                   life_time=-1.0)

        lane_width = self.map.get_waypoint(location).lane_width

        # start location
        # sequence as [+x -x +y -y] according to local rotation
        x_shift = -1. * lane_width
        x_slack = 0.75  # slack to fit different junction
        x_start = np.array([
            location.x + (extent.x + self.route_distance[0]),
            location.x - (extent.x + self.route_distance[0]),
            location.x + x_slack * lane_width + x_shift,
            location.x - x_slack * lane_width + x_shift,
        ])

        y_shift = 0. * lane_width
        y_slack = 0.75
        y_start = np.array([
            location.y - y_slack * lane_width + y_shift,
            location.y + y_slack * lane_width + y_shift,
            location.y + (extent.y + self.route_distance[0]),
            location.y - (extent.y + self.route_distance[0]),
        ])

        # plot the fixed coord center, for locate correct lane
        loc = carla.Location(x=location.x + x_shift,
                             y=location.y
                             )

        trans = carla.Transform(loc, rotation)
        draw_waypoint(self.world, trans, color=(white, white))

        start_wapoints = []
        for i in range(4):
            start_location = carla.Location(x=x_start[i], y=y_start[i], z=0)
            start_waypoint = self.map.get_waypoint(location=start_location,
                                                   project_to_road=True,  # not in the center of lane(road)
                                                   lane_type=carla.LaneType.Driving)

            lane_id = start_waypoint.lane_id

            start_wapoints.append(start_waypoint)
            draw_waypoint(self.world, start_waypoint)
            print('start waypoint is plot. coord: [',
                  start_waypoint.transform.location.x,
                  start_waypoint.transform.location.y,
                  start_waypoint.transform.location.z,
                  ']'
                  )

        return start_wapoints

    def generate_route(self, start_waypoint, turn_flag=-1):
        """
        Generate a route by given start waypoint.

        param turn_flag: left -1, straight 0, right 1
        """
        # 1. get key waypoint, exit point of the junction
        # get exit waypoint of the junction, with a turn flag
        exit_waypoint = generate_target_waypoint(waypoint=start_waypoint, turn=turn_flag)
        draw_waypoint(self.world, exit_waypoint, color=(blue, yellow))

        end_waypoint, _ = get_waypoint_in_distance(exit_waypoint, self.route_distance[1])
        draw_waypoint(self.world, end_waypoint, color=(magenta, yan))

        # additional method to return the complete list of the route in junction
        # return format: (list([waypoint, RoadOption]...), destination_waypoint)
        # wp_list, target_waypoint = generate_target_waypoint_list(waypoint=start_wapoints[2], turn=-1)

        # 2. generate a route by the keypoint
        # this method requires list of carla.Location format
        waypoint_list = [start_waypoint.transform.location,
                         exit_waypoint.transform.location,
                         end_waypoint.transform.location,
                         ]
        gps_route, route = interpolate_trajectory(world=self.world,
                                                  waypoints_trajectory=waypoint_list,
                                                  hop_resolution=1.0)

        # plot generated route
        for i, item in enumerate(route):
            trans = item[0]
            draw_waypoint(self.world, trans)

        return route

    def run(self):
        """"""
        # junction
        junction = self.get_junction_by_location(junction_center)

        start_waypoints = self.get_start_waypoints(junction)

        route = self.generate_route(start_waypoint=start_waypoints[2])

        return route


def generate_route():

    env = RouteGenerator()
    env.set_spectator_overhead(junction_center)

    env.get_junction_by_location(junction_center)

    start_waypoints = env.get_start_waypoints(env.junction)

    route = env.generate_route(start_waypoint=start_waypoints[2])

    return route


def generate_route_2(carla_api):
    """
    todo save the path in a script. Load the route before the training.
    """
    route_gen = RouteGenerator_2(carla_api)
    route = route_gen.run()

    return route


def test_loop():

    env = BasicEnv()
    env.set_spectator_overhead(junction_center)

    # get_waypoint(self, location, project_to_road=True, lane_type=carla.LaneType.Driving)
    waypoint = env.map.get_waypoint(start_location)

    road_id = waypoint.road_id
    section_id = waypoint.section_id
    lane_id = waypoint.lane_id

    lane_width = waypoint.lane_width

    print('')


if __name__ == '__main__':
    # test_loop()

    route = generate_route()

    print('')
