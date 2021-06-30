"""
A developing class to store information of a traffic flow.

Following information are supposed to be included:

 - index of the route, spawn point, sink location, turn flag and route
 -

todo todolist
 - use a data structure to store info about traffic flows
 - get necessary info from other modules

"""


class TrafficFlow:
    """
    Management of a traffic flow.
    """
    # some public info
    traffic_flows = {
        'positive_x': {
            'spawn_transforms': [],
            'sink_location': [],
        },
        'negative_x': {
            'spawn_transforms': [],
            'sink_location': [],
        },
        'positive_y_0': {
            'spawn_transforms': [],
            'sink_location': [],
        },
        'positive_y_1': {
            'spawn_transforms': [],
            'sink_location': [],
        },
        'negative_y_0': {
            'spawn_transforms': [],
            'sink_location': [],
        },
        'negative_y_1': {
            'spawn_transforms': [],
            'sink_location': [],
        },
    }

    def __init__(self, index: int):
        self.index = index

    def set_traffic_flow_params(self):
        """
        Setter method to set params for each traffic flow.
        """
        # distance between vehicles
        # spawn_distance = ()
        # target_speed = ()
        pass

    def get_tf_info_dict(self, index):
        """
        Get the info dict of a single traffic flow.
        """
        tf_dict = {}

        # determine turn_flag
        if index in [0, 5, 7, 12]:
            turn_flag = -1
        elif index in [1, 3, 6, 8, 10, 13]:
            turn_flag = 0
        elif index in [2, 4, 9, 11]:
            turn_flag = 1
        else:
            raise ValueError('Index number exceeds.')

        # determine spawn transform
        if index in [0, 1, 2]:
            spawn_transforms = self.traffic_flows['positive_x']['spawn_transforms']
        elif index in [3, 4]:
            spawn_transforms = self.traffic_flows['positive_y_1']['spawn_transforms']
        elif index in [5, 6]:
            spawn_transforms = self.traffic_flows['positive_y_0']['spawn_transforms']
        elif index in [7, 8, 9]:
            spawn_transforms = self.traffic_flows['negative_x']['spawn_transforms']
        elif index in [10, 11]:
            spawn_transforms = self.traffic_flows['negative_y_1']['spawn_transforms']
        elif index in [12, 13]:
            spawn_transforms = self.traffic_flows['negative_y_0']['spawn_transforms']
        else:
            raise ValueError('Index number exceeds.')

        # determine sink location
        if index in [0, 13]:
            sink_location = self.traffic_flows['negative_y_0']['sink_location']
        elif index in [1, 5, 11]:
            sink_location = self.traffic_flows['positive_x']['sink_location']
        elif index in [2, 3]:
            sink_location = self.traffic_flows['positive_y_1']['sink_location']
        elif index in [4, 8, 12]:
            sink_location = self.traffic_flows['negative_x']['sink_location']
        elif index in [6, 7]:
            sink_location = self.traffic_flows['positive_y_0']['sink_location']
        elif index in [9, 10]:
            sink_location = self.traffic_flows['negative_y_1']['sink_location']
        else:
            raise ValueError('Index number exceeds.')

        return tf_dict
