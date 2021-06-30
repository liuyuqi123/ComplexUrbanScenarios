"""
Traffic light state Controller.

Manually set the duration time of each phase of the traffic light.
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

# carla module
from train.gym_carla.envs.BasicEnv import BasicEnv
from train.gym_carla.modules.carla_module import CarlaModule
from train.gym_carla.util_development.util_visualization import draw_waypoint

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)

junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)


class TrafficLightsController(CarlaModule):
    """
    Control the tls of a certain junction.

    todo this module is supposed to be corresponded with traffic_flow module
    """

    # phase_time refers to the duration time of each state of traffic lights on x direction
    # notice the format
    phase_time = {
        'x_green_phase': 15.,  # ref: 19.
        'x_yellow_phase': 3.,
        'y_green_phase': 15.,
        'y_yellow_phase': 3.,
    }
    """
    An example of different duration time on x and y direction
    'x_green_phase': 19.,
    'x_yellow_phase': 5.,
    'y_green_phase': 8.,
    'y_yellow_phase': 2.,
    
    ==> refers that x_red_phase=10.(8+2), y_red_phase=24.(19+5)
    
    """

    # tl state
    green_state = carla.TrafficLightState.Green
    yellow_state = carla.TrafficLightState.Yellow
    red_state = carla.TrafficLightState.Red

    tl_logic = {
        'phase_0': {
            'duration': phase_time['x_green_phase'],
            'positive_x': green_state,
            'negative_x': green_state,
            'positive_y': red_state,
            'negative_y': red_state,
        },
        'phase_1': {
            'duration': phase_time['x_yellow_phase'],
            'positive_x': yellow_state,
            'negative_x': yellow_state,
            'positive_y': red_state,
            'negative_y': red_state,
        },
        'phase_2': {
            'duration': phase_time['y_green_phase'],
            'positive_x': red_state,
            'negative_x': red_state,
            'positive_y': green_state,
            'negative_y': green_state,
        },
        'phase_3': {
            'duration': phase_time['y_yellow_phase'],
            'positive_x': red_state,
            'negative_x': red_state,
            'positive_y': yellow_state,
            'negative_y': yellow_state,
        },
    }

    def __init__(self, carla_api, junction: carla.Junction = None):

        super(TrafficLightsController, self).__init__(carla_api)

        self.debug = True

        # todo improve API, using parent class
        if junction:
            self.junction = junction
        else:  # use the default junction in Town03
            self.junction = self.get_junction_by_location(junction_center)

        self.traffic_lights = []
        self.tl_dict = {}

        # timestamp that last time tl changes
        self.last_change_timestamp = None

        # 4 phases in total, 'phase_'+index
        self.current_phase_index = int(0)  # default initial state

        # print current phase time
        print("Current traffic lights setting: ")
        for key, item in self.phase_time.items():
            print(key, ': ', item, 'seconds')

    def __call__(self, *args, **kwargs):
        """
        The module needs to be called and ticked by Env each timestep.

        todo fix the api
        """
        self.run_step()

    def run_step(self):
        """
        This method will be ticked each timestep with the RL env.

        This method is supposed to be called after world tick.
        """

        # get current timestamp
        timestamp = self.world.get_snapshot().timestamp
        t_1 = timestamp.elapsed_seconds  # current time

        # duration time of each state
        t_0 = self.last_change_timestamp.elapsed_seconds  # latest time

        # elapsed time since last change
        elapsed_time = t_1 - t_0

        # current information of the
        # tl_info = self.get_tl_info()

        # duration time of current phase
        phase_name = 'phase_' + str(self.current_phase_index)
        phase_duration = self.tl_logic[phase_name]['duration']

        # if time is reached
        if elapsed_time >= phase_duration:
            # shift to next state
            if self.current_phase_index == int(3):
                self.current_phase_index = 0
            else:
                self.current_phase_index = int(self.current_phase_index + 1)

            # set the traffic light to next phase
            self.set_tl_state(self.current_phase_index)

            # reset timer
            self.last_change_timestamp = timestamp

    def set_tl_state(self, phase_index=int(0)):
        """
        Reset all traffic lights state to a given state phase

        param: phase_index: int, index from (0, 1, 2, 3), refers to the state index of tl on x direction.
        """
        # get phase name
        self.current_phase_index = phase_index
        phase_name = 'phase_' + str(self.current_phase_index)

        # clear the elapsed time
        self.traffic_lights[0].reset_group()

        for label, tl in self.tl_dict.items():
            tl.freeze(True)
            state = self.tl_logic[phase_name][label]
            tl.set_state(state)

        # update timestamp of latest change
        self.last_change_timestamp = self.world.get_snapshot().timestamp

    def assign_tl_id(self):
        """
        Assign junction traffic lights id, determine relative location.

        Relative location is determined by the trigger volume.

        :return:
        """
        loc = np.array([[0, 0]])

        for tl in self.traffic_lights:
            # location of the traffic light
            tl_location = tl.get_location()
            # transform of the tl
            transform = tl.get_transform()

            # transform matrix from Actor system to world system
            _T = transform.get_matrix()
            trans_world_actor = np.array([[_T[0][0], _T[0][1], _T[0][2]],
                                          [_T[1][0], _T[1][1], _T[1][2]],
                                          [_T[2][0], _T[2][1], _T[2][2]]])

            # location of the trigger volume
            trigger_volume = tl.trigger_volume
            # relative location in Actor coord system
            _trigger_volume_rel_loc = trigger_volume.location
            _trigger_volume_rel_loc = np.array([[_trigger_volume_rel_loc.x],
                                                [_trigger_volume_rel_loc.y],
                                                [_trigger_volume_rel_loc.z]])
            # trigger volume location relative to Actor in world coord system
            trigger_volume_rel_loc = np.dot(trans_world_actor, _trigger_volume_rel_loc)
            trigger_volume_rel_loc = np.squeeze(trigger_volume_rel_loc)
            trigger_volume_relative_location = carla.Location(x=trigger_volume_rel_loc[0],
                                                              y=trigger_volume_rel_loc[1],
                                                              z=trigger_volume_rel_loc[2])
            # trigger volume location in global coord system
            trigger_volume_center = tl_location + trigger_volume_relative_location

            # location relative to junction center
            # if using junction bounding box center coords
            junc_loc = self.junction.bounding_box.location
            relative_loc = trigger_volume_center - junc_loc

            rel_loc = np.array([[relative_loc.x, relative_loc.y]])
            loc = np.concatenate((loc, rel_loc), axis=0)

        # todo check if this is correct
        # index means which incoming traffic flow is controlled
        # determine index in tl_list
        pos_x_index = np.argmax(loc, axis=0)[0]
        pos_y_index = np.argmax(loc, axis=0)[1]
        neg_x_index = np.argmin(loc, axis=0)[0]
        neg_y_index = np.argmin(loc, axis=0)[1]

        # tl_dict stores traffic light actors
        self.tl_dict = {
            'positive_x': self.traffic_lights[pos_x_index - 1],  # the first row of loc is (0, 0)
            'negative_x': self.traffic_lights[neg_x_index - 1],
            'positive_y': self.traffic_lights[pos_y_index - 1],
            'negative_y': self.traffic_lights[neg_y_index - 1],
        }

    def get_traffic_lights(self):
        """
        Get traffic lights of a given junction.

        The tl actors is stored in a dict(label as the relative location) and a list.
        """
        # get all the tl in the map
        actor_list = self.world.get_actors()
        tl_list = actor_list.filter('traffic.traffic_light')
        if tl_list:
            # trans to list
            tl_list = [x for x in tl_list]
        else:
            raise ValueError('Fail to acquire TL.')

        # manipulate traffic light and visualize it
        for tl in tl_list:

            # get the tl in the certain junction
            transform = tl.get_transform()
            tl_loc = tl.get_location()

            # if the tl is in the junction
            junc_bbox = self.junction.bounding_box
            in_junction = \
                (tl_loc.x <= junc_bbox.location.x + junc_bbox.extent.x) and \
                (tl_loc.x >= junc_bbox.location.x - junc_bbox.extent.x) and \
                (tl_loc.y <= junc_bbox.location.y + junc_bbox.extent.y) and \
                (tl_loc.y >= junc_bbox.location.y - junc_bbox.extent.y)

            if not in_junction:
                # set irrelevant traffic lights green to avoid traffic jam
                tl.set_green_time(999999.)
                tl.set_state(carla.TrafficLightState.Green)
                tl.freeze(True)
                # must tick to take effect
                self.world.tick()
                continue

            self.traffic_lights.append(tl)

            # plot the traffic light by its transform
            if self.debug:
                _transform = transform
                _transform.location.z = 1.0
                draw_waypoint(self.world, _transform)  # draw location
                # set_spectator_overhead(self.world, _transform.location)

            # plot area in which traffic light will effect
            bbox = tl.trigger_volume  # carla.BoundingBox
            # bounding box
            # self.debug_helper.draw_box(box=bbox,
            #                            rotation=bbox.rotation,
            #                            thickness=0.15,
            #                            color=red,
            #                            life_time=-1.0)

            # local_vertices = bbox.get_local_vertices()  # relative location to the traffic light
            world_vertices = bbox.get_world_vertices(transform)
            all_vertex = np.array([[0, 0, 0]])
            for ver_loc in world_vertices:
                # draw vertices of the bbox in the world
                # self.debug_helper.draw_point(ver_loc, size=0.1, color=blue, life_time=99999)

                # get vertices in ndarray
                _vertex = np.array([[ver_loc.x, ver_loc.y, ver_loc.z]])
                all_vertex = np.concatenate((all_vertex, _vertex), axis=0)

            all_vertex = np.delete(all_vertex, 0, 0)
            # draw 2D bbox for each selected traffic light
            # get 2D bbox of the bbox
            x_max_index = np.argmax(all_vertex, axis=0)[0]
            y_max_index = np.argmax(all_vertex, axis=0)[1]
            x_min_index = np.argmin(all_vertex, axis=0)[0]
            y_min_index = np.argmin(all_vertex, axis=0)[1]
            z_max_index = np.argmax(all_vertex, axis=0)[2]

            _x_max = all_vertex[x_max_index][0]  # last index refers to x, y, z
            _y_max = all_vertex[y_max_index][1]
            _x_min = all_vertex[x_min_index][0]
            _y_min = all_vertex[y_min_index][1]
            z = all_vertex[z_max_index][2]

            # connect vertices and draw lines
            bbox_vertives = [
                carla.Location(_x_max, _y_max, z),
                carla.Location(_x_min, _y_max, z),
                carla.Location(_x_min, _y_min, z),
                carla.Location(_x_max, _y_min, z),
                carla.Location(_x_max, _y_max, z),  # duplicated (x_max, y_max)
            ]

            for i in range(4):
                self.debug_helper.draw_line(bbox_vertives[i],
                                            bbox_vertives[i+1],
                                            thickness=0.15,
                                            color=magenta,
                                            life_time=99999,
                                            )

        # assign label for each tl
        # label is defined by the relative location to the junction center
        self.assign_tl_id()

        # set traffic lights to initial state
        self.set_tl_state()

        return self.tl_dict

    def set_tl_logic(self, phase_time: dict = None):
        """
        Set traffic lights control logic by resetting the phase time.

        This method is called only when you wish to reset the phase time.

        :param phase_time: new phase time
        """
        # update tl logic if phase time is changed
        if phase_time:
            self.phase_time['x_green_phase'] = phase_time['x_green_phase']
            self.phase_time['x_yellow_phase'] = phase_time['x_yellow_phase']
            self.phase_time['y_green_phase'] = phase_time['y_green_phase']
            self.phase_time['y_yellow_phase'] = phase_time['y_yellow_phase']

        # apply tl logic to traffic lights
        for tl in self.traffic_lights:
            tl.set_green_time(self.phase_time['green_phase'])
            tl.set_yellow_time(self.phase_time['yellow_phase'])
            red_phase = self.phase_time['green_phase'] + self.phase_time['yellow_phase']
            tl.set_red_time(red_phase)

    def get_phase_index(self):
        """
        A Getter method.
        Get tls phase index of current timestep.

        :return: int, phase index
        """

        return self.current_phase_index

    def get_tl_info(self):
        """
        Retrieve traffic light information of current timestep

        :return: dict: tl_info
        """
        _tl_info_tuple = {
            'tl_actor': None,
            'current_state': None,
            'elapsed_time': None,
            'green_time': None,
            'red_time': None,  # must equal to green time
            'yellow_time': None,
        }

        tl_info = {
            'positive_x': None,
            'negative_x': None,
            'positive_y': None,
            'negative_y': None,
        }

        for item in self.tl_dict.items():
            tl_name = item[0]
            tl = item[1]

            tl_info_tuple = {
                'tl_actor': tl,
                'pole_index': tl.get_pole_index(),
                'current_state': tl.state,
                'elapsed_time': tl.get_elapsed_time(),
                'green_time': tl.get_green_time(),
                'red_time': tl.get_red_time(),
                'yellow_time': tl.get_yellow_time(),
            }

            # store info in dict
            tl_info[tl_name] = tl_info_tuple

        return tl_info

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

        # plot bbox
        bbox = junction.bounding_box
        self.debug_helper.draw_box(box=bbox,
                                   rotation=bbox.rotation,
                                   thickness=0.25,
                                   color=red,
                                   life_time=-1.0)

        # set the spectator
        # set_spectator_overhead(self.world, bbox.location, h=50)

        return junction

    def test_api(self):
        """
        This method is supposed to test all available API of the carla.TrafficLight

        params: tl is a carla.TrafficLight instance.
        """
        for tl in self.traffic_lights:
            print('tl: ', tl)

            # ----------   attributes   ----------
            state = tl.state

            # ----------     methods   ----------
            is_frozen = tl.is_frozen()
            # getters
            pole_index = tl.get_pole_index()
            group_traffic_lights = tl.get_group_traffic_lights()
            state = tl.get_state()
            elapsed_time = tl.get_elapsed_time()
            # state
            green_time = tl.get_green_time()
            red_time = tl.get_red_time()
            yellow_time = tl.get_yellow_time()
            # Setters
            tl.set_green_time(3.)
            tl.set_red_time(5.)
            tl.set_yellow_time(7.)

            """
            How to init a TrafficLightState:
            
            Optional traffic light state:
            Red
            Yellow
            Green
            Off
            Unknown
            """
            # set state for the traffic light actor
            target_state = carla.TrafficLightState.Green
            tl.set_state(target_state)

            """
            Reset the state of the whole group.
            Doc: Resets the state of the traffic lights of the group 
            to the initial state at the start of the simulation
            """
            tl.reset_group()


def test_tl_change():
    """
    Test how traffic light state is changed.

    The traffic light of same group will change in a cyclic pattern,
    in orders of pole index.

    The set_time method will take effect after world tick.
    """

    # create a carla env
    env = BasicEnv(traffic_manager_port=int(8100))
    # env.set_world(sync_mode=False)  # for debug

    tls = TrafficLightsController(env.get_env_api())
    # get traffic light of the certain junction
    tls.get_traffic_lights()

    print('-' * 25)
    tl_info = tls.get_tl_info()
    for key, info_dict in tl_info.items():
        print(key, ":")
        current_state = info_dict['current_state']
        elapsed_time = info_dict['elapsed_time']
        print('current_state: ', current_state)
        print('elapsed_time: ', elapsed_time)

    tls.traffic_lights[0].reset_group()

    # test reset tl_logic
    tl_logic = {
        'x_green_phase': 111.,
        'x_yellow_phase': 55.,
        'y_green_phase': 111.,
        'y_yellow_phase': 55.,
    }
    tls.set_tl_logic(tl_logic)

    # env.set_world(sync_mode=False)

    tls.world.tick()
    timestamp = env.world.get_snapshot().timestamp
    elapsed_seconds = timestamp.elapsed_seconds
    print('t1: ', elapsed_seconds)

    # test state shifting
    print('-' * 25)
    tl_info = tls.get_tl_info()
    aaa = tl_info
    for key, info_dict in tl_info.items():
        print(key, ":")
        current_state = info_dict['current_state']
        elapsed_time = info_dict['elapsed_time']
        print('current_state: ', current_state)
        print('elapsed_time: ', elapsed_time)

    print('-' * 25)
    # env.set_world(sync_mode=False)
    timestamp = env.world.get_snapshot().timestamp
    elapsed_seconds = timestamp.elapsed_seconds
    print('t2: ', elapsed_seconds)

    while True:
        tls.world.tick()
        # env.set_world(sync_mode=False)
        timestamp = env.world.get_snapshot().timestamp
        elapsed_seconds = timestamp.elapsed_seconds

        tl_info = tls.get_tl_info()
        bbb = tl_info
        # for key, info_dict in tl_info.items():
        #     print(key, ":")
        #     current_state = info_dict['current_state']
        #     elapsed_time = info_dict['elapsed_time']
        #     print('current_state: ', current_state)
        #     print('elapsed_time: ', elapsed_time)

        for label, info_dict in bbb.items():
            if info_dict['current_state'] is not aaa[label]['current_state']:
                print('-' * 25)
                for key, _info_dict in bbb.items():
                    print(key, ":")
                    current_state = _info_dict['current_state']
                    elapsed_time = _info_dict['elapsed_time']
                    print('current_state: ', current_state)
                    print('elapsed_time: ', elapsed_time)

                aaa = bbb
                print('t3: ', elapsed_seconds)


if __name__ == '__main__':
    # test how tl state change
    test_tl_change()
