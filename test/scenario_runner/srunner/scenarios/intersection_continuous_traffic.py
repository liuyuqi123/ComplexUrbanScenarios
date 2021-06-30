#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a left turn
"""

from six.moves.queue import Queue  # pylint: disable=relative-import

import numpy as np
import py_trees

import carla

from agents.navigation.local_planner import RoadOption

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    ActorDestroy,
    # ActorSource,
    # ActorSink,
    # WaypointFollower
)

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance

from srunner.drl_code.scenario_utils.basic_scenario_fixed import (
    BasicScenario,
)

from srunner.drl_code.scenario_utils.atomic_behaviors_fixed import (
    ActorSource,
    ActorSink,
    WaypointFollower,
)
from srunner.drl_code.scenario_utils.atomic_criteria_fixed import (
    CollisionTest,
    RouteCompletionTest,
)

from srunner.drl_code.scenario_utils.util_visualization import set_spectator_overhead


# default junction center
junction_location = carla.Location(x=-1.32, y=132.69, z=0.00)


class IntersectionContinuousTraffic(BasicScenario):
    """
    todo add api to set params of the scenario

    This class describe scenarios from TCMAX,
    in which ego vehicle encounter a continuous traffic flow.

    Some additional args and params are added to generation of the validation.

    Implementation class for Hero
    Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.

    This is a single ego vehicle scenario
    """

    # Timeout of scenario in seconds
    timeout = 60.

    def __init__(self,
                 world,
                 ego_vehicles,
                 config,
                 randomize=False,
                 debug_mode=False,
                 criteria_enable=True,
                 # timeout=80,
                 # =========================
                 # additional args
                 ego_route=None,  # ego route for success check
                 wp_plan=None,  # waypoints route of traffic flow
                 target_speed=None,  # target speed of traffic flow
                 distance_gap=None,  # distance between traffic flow spawning
                 verbose=True,  # visualize everything
                 ):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self._brake_value = 0.5  # todo what for??
        self._ego_distance = 60.
        self._traffic_light = None
        self._other_actor_transform = None
        self._blackboard_queue_name = 'IntersectionContinuousTraffic/actor_flow_queue'
        self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
        self._initialized = True

        # ================   additional params   ================
        set_spectator_overhead(self._world, junction_location, yaw=270, h=70)

        # ----------   traffic flow settings   ----------
        # turning option of traffic flow, Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
        self.wp_plan = wp_plan
        self.target_velocity = target_speed
        self.distance_gap = distance_gap

        # randomize distance gap
        self.randomize = randomize

        # route of ego vehicle
        self.ego_route = ego_route

        # visualize everything
        self.verbose = verbose

        # whether enable collision detection of NPC vehicles
        self.avoid_collision = False

        # todo add a setter for this
        # self.timeout = timeout

        # ========================================================

        super(IntersectionContinuousTraffic, self).__init__("IntersectionContinuousTraffic",
                                                            ego_vehicles,
                                                            config,
                                                            world,
                                                            debug_mode,
                                                            criteria_enable=criteria_enable,
                                                            terminate_on_failure=True,
                                                            )

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        # todo set all traffic lights green
        if self._traffic_light is None or traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z - 500),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, self._other_actor_transform)
        first_vehicle.set_transform(first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is turning left in an urban area,
        at a signalized intersection, while other actor coming straight
        .The hero actor may turn left either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        sequence = py_trees.composites.Sequence("Sequence Behavior")

        # adding flow of actors
        actor_source = ActorSource(
            actor_type_list=['vehicle.tesla.model3'],
            transform=self._other_actor_transform,
            threshold=self.distance_gap,
            blackboard_queue_name=self._blackboard_queue_name,
            actor_limit=99999,
            name="ActorSource",
            # ======== additional args ========
            init_speed=self.target_velocity,  # initial speed newly spawned vehicle, km/h
            randomize=self.randomize,
        )

        # destroying flow of actors
        actor_sink = ActorSink(self.wp_plan[-5][0].transform.location,  # don't use the last waypoint
                               1.5,  # threshold to destroy any vehicle
                               )

        # follow waypoints until next intersection
        move_actor = WaypointFollower(
            self.other_actors[0],
            self.target_velocity / 3.6,
            plan=self.wp_plan,
            blackboard_queue_name=self._blackboard_queue_name,
            avoid_collision=self.avoid_collision,
        )

        # wait
        wait = DriveDistance(self.ego_vehicles[0], self._ego_distance)  # todo does this lead to success termination

        # Behavior tree
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # root.add_child(wait)
        root.add_child(actor_source)
        root.add_child(actor_sink)
        root.add_child(move_actor)

        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(root)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criteria = CollisionTest(self.ego_vehicles[0],
                                           # terminate_on_failure=True,
                                           )

        route_completion_criteria = RouteCompletionTest(actor=self.ego_vehicles[0],
                                                        route=self.ego_route,
                                                        # terminate_on_failure=True,
                                                        )

        criteria.append(collision_criteria)
        criteria.append(route_completion_criteria)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()

