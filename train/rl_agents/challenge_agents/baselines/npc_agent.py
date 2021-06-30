"""
A fixed NPC agent with API to set route.
"""

import carla
from agents.navigation.basic_agent import BasicAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from train.gym_carla.util_development.util_visualization import draw_waypoint


class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    # def __init__(self):
    #
    #     self.route = None

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self._route_assigned = False
        self._agent = None

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}


        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        """

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor)

            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))

                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True

        else:
            control = self._agent.run_step()
                
        """

        control = self._agent.run_step()

        return control

    def set_route(self, wp_route):
        """
        Set complete route for agent.

        todo merge this method with original set global plan
        """
        # plot generated route
        for wp, _ in wp_route:
            transform = wp.transform
            draw_waypoint(self.world, transform)

        self.route = wp_route

    def set_ego_vehicles(self, ego_vehicle):

        # self.ego_vehicle = ego_vehicle

        self._agent = BasicAgent(ego_vehicle)

        # todo add API on basic agent, hazard_detected
        if self.route:
            self._agent._local_planner.set_global_plan(self.route)
        else:
            raise ValueError('Route is not found, please check.')














