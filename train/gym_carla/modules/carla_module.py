"""
This is a parent class of carla module.

Any developed carla module is supposed to inherit this class.
"""

import carla


class CarlaModule:
    """
    Basic carla module class.

    This class is responsible to manage a function of a simulation.

    A module is initialized by a created carla env.
    """

    def __init__(self, carla_api):
        self.carla_api = carla_api
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        self.npc_vehicles = []
        self.ego_vehicle = None

    def reload_carla_world(self, carla_api):
        """
        Reload carla world by reset carla_api.
        """
        self.carla_api = carla_api

        # only update world related attributes
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']

    def __call__(self, *args, **kwargs):
        """
        The module will be called each timestep when carla world ticks.
        """
        pass

    def run_step(self):
        """
        Do a run step to tick all necessary step.
        """
        pass

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

    def update_vehicles(self):
        """
        Update vehicles of current timestep.
        This method is supposed to be called each timestep

        :return:
        """
        self.npc_vehicles = []
        self.ego_vehicle = None

        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # carla.Actorlist, iterable

        if vehicle_list:
            for veh in vehicle_list:  # veh is carla.Vehicle
                attr = veh.attributes  # dict
                role_name = attr['role_name']
                # filter ego vehicle
                if role_name in ['ego', 'hero']:
                    self.ego_vehicle = veh
                else:
                    self.npc_vehicles.append(veh)
