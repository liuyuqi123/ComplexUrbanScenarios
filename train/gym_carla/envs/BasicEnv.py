"""
This is a basic class to create a carla env.
"""

# ==================================================
# import carla module

from train.gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

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

import argparse
import logging
import time

import heapq
from datetime import datetime

import random
import numpy as np
import math
import traceback

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)


class BasicEnv:
    """
    The basic class to generate a carla env for test.
    """
    def __init__(self,
                 host='localhost',
                 port=int(2000),
                 tm_port=int(8000),
                 town='Town03',
                 client_timeout=100.0,
                 timestep=0.05,
                 frequency=None,
                 sync_mode=True,
                 ):
        # setup client
        self.client = carla.Client(host, port)
        self.client.set_timeout(client_timeout)
        self.map_name = town  # str, name of the map
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.debug_helper = self.world.debug  # world debug for plot
        self.blueprint_library = self.world.get_blueprint_library()  # blueprint

        # frequency has priority
        if frequency:
            self.timestep = 1 / frequency
        else:
            self.timestep = timestep

        self.sync_mode = sync_mode
        self.set_world(sync_mode=sync_mode)  # world settings
        self.set_weather()  # weather

        self.spectator = self.world.get_spectator()
        self.traffic_manager = self.client.get_trafficmanager(tm_port)  # check carla version before using

        # vehicles information
        self.ego_vehicle = None
        # if there are multiple ego vehicles
        self.ego_vehicles = []
        # vehicles except ego vehicle
        self.npc_vehicles = []

    def reload_carla_world(self, reset_settings=False):
        """
        todo carla.Client.reload_world() method is fixed in 0.9.11 version.
         we can keep previous world settings with an arg reset_settings(False)

        Reload carla world.
        """
        self.world = self.client.load_world(self.map_name)
        # todo fix api with newer version of carla, add all setting params to args
        # set world with previous settings
        self.set_world(sync_mode=self.sync_mode)

        # update world related attributes
        self.map = self.world.get_map()
        self.debug_helper = self.world.debug
        self.blueprint_library = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()

    def get_env_api(self):
        """
        Get carla environment management API.
        """
        carla_management = {
            'client': self.client,
            'world': self.world,
            'map': self.map,  # carla.Map
            'debug_helper': self.debug_helper,
            'blueprint_library': self.blueprint_library,
            'spectator': self.spectator,
            'traffic_manager': self.traffic_manager,
        }

        return carla_management

    def set_world(self, sync_mode=True, no_render_mode=False):
        """
        Setup carla world settings.
        Under sync_mode(sync_mode = True), require world.tick() to run
        """
        settings = self.world.get_settings()
        # world settings parameters
        settings.fixed_delta_seconds = self.timestep
        settings.no_rendering_mode = no_render_mode
        settings.synchronous_mode = sync_mode
        self.world.apply_settings(settings)
        self.world.tick()  # refresh world

    def set_weather(self, weather='ClearNoon'):
        """
        Set weather for the world
        Common weather in carla:
            ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon,
            SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset,
            CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset,
            MidRainSunset, HardRainSunset.
        """
        weather_dict = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
            'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
            'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
            'MidRainSunset': carla.WeatherParameters.MidRainSunset,
            'HardRainSunset': carla.WeatherParameters.HardRainSunset,
        }
        weather_selection = None
        for key in weather_dict:
            if key == weather:
                weather_selection = weather_dict[key]
        if not weather_selection:
            print('Specified weather not found. ClearNoon is set.')
            weather_selection = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather_selection)
        self.world.tick()

    def set_spectator_actor(self, actor):
        """Set behind view on an actor"""
        transform = actor.get_transform()
        # location = transform.location
        # rotation = transform.rotation

        # behind distance - d, height - h
        _d = 8
        _h = 6
        angle = transform.rotation.yaw
        a = math.radians(180 + angle)
        location = carla.Location(x=_d * math.cos(a),
                                  y=_d * math.sin(a),
                                  z=_h) + transform.location
        rotation = carla.Rotation(yaw=angle, pitch=6)

        self.spectator.set_transform(carla.Transform(location, rotation))
        self.world.tick()
        print("Spectator is set to behind view.")

    def set_spectator_overhead(self, location, yaw=0, h=50):
        """
        Set spectator from an overview.

        param location: location of the spectator
        param h(float): height of spectator when using the overhead view
        """

        height = h
        # height = 100
        location = carla.Location(0, 0, height) + location
        rotation = carla.Rotation(yaw=yaw, pitch=-90)  # rotate to forward direction

        self.spectator.set_transform(carla.Transform(location, rotation))
        self.world.tick()

        print("Spectator is set to overhead view.")

    def draw_waypoint(self, transform, color=(red, green)):
        """
        Draw a point determined by transform(or waypoint).
        A spot to mark location
        An arrow to x-axis direction vector

        :param transform: carla.Transform or carla.Waypoint
        :param color: color of arrow and spot
        """
        if isinstance(transform, carla.Waypoint):
            transform = transform.transform
        scalar = 1.5
        yaw = np.deg2rad(transform.rotation.yaw)
        vector = scalar * np.array([np.cos(yaw), np.sin(yaw)])
        start = transform.location
        end = start + carla.Location(x=vector[0], y=vector[1], z=0)
        # plot the waypoint
        self.debug_helper.draw_point(start, size=0.05, color=color[0], life_time=99999)
        self.debug_helper.draw_arrow(start, end, thickness=0.25, arrow_size=0.20, color=color[1], life_time=99999)
        self.world.tick()

    def update_vehicles(self):
        """
        Update existing vehicles in current world.
        """
        self.npc_vehicles = []
        self.ego_vehicle = None

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a actorlist instance, iterable

        if vehicle_list:
            for veh in vehicle_list:
                attr = veh.attributes  # dict
                # filter ego vehicle by role name
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':
                    self.ego_vehicle = veh
                else:
                    self.npc_vehicles.append(veh)

            # print('All vehicles are updated.')
            # if not self.ego_vehicle:
            #     print('ego vehicle not found.')

    @staticmethod
    def coord2loc(coords):
        """
        Transform a coords to carla location.
        Coords will be set default on the ground, z=0.
        :param coords: coordinates of np.array of list.
        :return: carla.location
        """
        location = carla.Location(x=float(coords[0]), y=float(coords[1]), z=0.0)
        return location

    @staticmethod
    def loc2coord(location):
        """
        Transform a carla location to coords.
        :param location: carla.Location
        :return: coordinate in carla world.
        """
        coords = np.array([location.x, location.y, location.z])
        return coords


def test():

    env = BasicEnv()
    env.set_world(sync_mode=False)

    # jucntion in Town03
    junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

    env.set_spectator_overhead(junction_center, h=75)

    print('test env is created.')


if __name__ == '__main__':

    test()

