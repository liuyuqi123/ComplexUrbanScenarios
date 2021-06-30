"""
Some methods to help with scenario validation.

Reference on junction_route_generator

"""

import numpy as np
import math

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# todo check if need to modify this
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           get_waypoint_in_distance,
                                           )

from srunner.drl_code.scenario_utils.util_visualization import draw_waypoint
from srunner.drl_code.scenario_utils.carla_color import *


def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
    This method is fixed based on original interpolate_trajectory

    Given some raw keypoints interpolate a full dense trajectory to be used by the user.

    :param world: an reference to the CARLA world so we can use the planner
    :param waypoints_trajectory: the current coarse trajectory
    :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    :return: the full interpolated route both in waypoints.
    """

    dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []  # waypoints
    for i in range(len(waypoints_trajectory) - 1):  # Goes until the one before the last.

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0], wp_tuple[1]))

    return route


class ScenarioRouteManager:
    """
    This class help with management of a scenario route.
    """

    def __init__(self,
                 world: carla.World,
                 spawn_location: carla.Location,
                 debug=True,
                 verbose=True,
                 ):
        """
        A route generator for ego vehicle.

        todo improve this method to generate route for random vehicle.
        :param world:
        :param spawn_location:
        """
        self.world = world
        self.map = self.world.get_map()
        self.debug_helper = self.world.debug

        # get junction by ego spawn waypoint
        self.junction = None
        spawn_waypoint = self.map.get_waypoint(spawn_location,
                                               project_to_road=True,  # must set to True, center of lane
                                               )
        self.get_junction_by_route(spawn_waypoint)

        # # route in list of tuple
        # self.transform_route = []
        # self.waypoint_route = []
        # self.location_route = []

        # debug
        self.debug = debug
        # verbose option to visualize everything
        self.verbose = verbose

    def get_junction_by_location(self):
        """
        """
        pass

    def get_junction_by_route(self, start_waypoint):
        """
        Get junction in front of ego vehicle by exploring route.
        """
        # get start waypoint
        waypoint = start_waypoint

        reached_junction = False
        sampling_radius = 1.

        while not reached_junction:
            wp_choice = waypoint.next(sampling_radius)
            waypoint = wp_choice[0]

            if waypoint.is_junction:
                reached_junction = True

        # get junction by waypoint
        self.junction = waypoint.get_junction()

        return self.junction

    def get_route(self,
                  spawn_location,
                  distance: float = 10.,
                  turning_flag=-1,  # left -> -1, straight -> 0, right -> 1
                  resolution: float = 1.):
        """
        Generate a route for the scenario.

        Route is determined by class attribute turning_flag.

        params:
        distance: distance after the junction
        resolution: distance gap between waypoint in route.
        """

        waypoint_route = []
        transform_route = []
        location_route = []

        spawn_waypoint = self.map.get_waypoint(spawn_location,
                                               project_to_road=True,  # must set to True, center of lane
                                               )
        # start waypoint: beginning waypoint of the route,
        start_waypoint = spawn_waypoint.next(3.0)[0]  # assuming that next waypoint is not in junction
        # exit_waypoint: the first waypoint after leaving the junction
        exit_waypoint = generate_target_waypoint(waypoint=start_waypoint, turn=turning_flag)
        # end_waypoint: end waypoint of the route
        end_waypoint, _ = get_waypoint_in_distance(exit_waypoint, distance)

        # list of carla.Location for route generation
        raw_route = [start_waypoint.transform.location,
                     exit_waypoint.transform.location,
                     end_waypoint.transform.location,
                     ]

        waypoint_route = interpolate_trajectory(world=self.world,
                                                waypoints_trajectory=raw_route,
                                                hop_resolution=resolution)

        # get route of transform
        for wp, road_option in waypoint_route:
            transform_route.append((wp.transform, road_option))

        # get route of location
        for wp, road_option in waypoint_route:
            location_route.append((wp.transform.location, road_option))

        # ====================   visualization   ====================
        key_waypoints = [start_waypoint,
                         exit_waypoint,
                         end_waypoint,
                         ]

        for wp in key_waypoints:
            draw_waypoint(self.world, wp, color=(red, red))

        if self.verbose:
            # visualize complete route
            for wp, _ in waypoint_route:
                draw_waypoint(self.world, wp, color=(magenta, magenta))

        return waypoint_route, location_route, transform_route
