"""
A developing version of traffic flow manager2.

Wish to:
 - Add sink point for each traffic flow.

"""

import carla

from gym_carla.util_development.scenario_helper_modified import generate_target_waypoint, get_waypoint_in_distance

from gym_carla.modules.trafficflow.traffic_flow_manager2 import TrafficFlowManager2


class TrafficFlowManager3(TrafficFlowManager2):
    # todo add new condition for deleting vehicles, add sink location

    # start transform of all available traffic flow

    def __init__(self,
                 carla_api,
                 junction=None,
                 active_tf_direction=None,
                 tm_seed=int(0),
                 debug=False,
                 ):
        
        super(TrafficFlowManager3, self).__init__(
            carla_api=carla_api,
            junction=junction,
            active_tf_direction=active_tf_direction,
            tm_seed=tm_seed,
            debug=debug,
        )

        # carla.Location list
        self.sink_locations = []

    # todo fix delete vehicle check
    #  add sinking method

    def get_sink_location(self):
        """
        Get sink point of different traffic flow.

        Only straight flag needs to be considered, because left and right routes are duplicated.
        """

        # get all available sink points
        for key, item in self.traffic_flow_info:
            transform = item['spawn_transform']
            location = transform.location
            start_waypoint = self.map.get_waypoint(location=location,
                                                   project_to_road=True,
                                                   )

            # the waypoint after junction
            exit_waypoint = generate_target_waypoint(
                waypoint=start_waypoint,
                turn=0,
            )

            # the end of the road
            end_waypoint = exit_waypoint.next(1.0)[0]
            while not end_waypoint.is_intersection:
                end_waypoint = end_waypoint.next(1.0)[0]  # end_waypoint refers to the end of the whole route
            # distance gap from the end of the road
            distance_gap = 3.
            end_waypoint = end_waypoint.previous(distance_gap)[0]

            # as the sink location
            end_location = end_waypoint.transform.location

            self.sink_locations.append(end_location)

