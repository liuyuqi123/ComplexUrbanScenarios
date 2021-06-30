"""
Some utilitiesfor visulization of carla experiment.
"""

from __future__ import print_function

import glob
import os
import sys

# using carla 095
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'  # carla egg

# using carla096
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_096/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_096/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_096/PythonAPI'

# using carla098
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla/agents")
carla_path = '/home/lyq/CARLA_simulator/CARLA_098/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import math
import numpy as np
import random

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)


def set_spectator_overhead(world, location, yaw=0, h=75):
    """
    Set spectator from an overview.

    param location: location of the spectator
    param h - height (float): height of spectator when using the overhead view
    """
    location = carla.Location(location.x, location.y, h)
    rotation = carla.Rotation(yaw=yaw, pitch=-90)  # rotate to forward direction

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(location, rotation))
    world.tick()

    print("Spectator is set to overhead view.")


def draw_waypoint(world, waypoint, color=(red, green)):
    """
    Draw a waypoint in carla server.

    params world: carla.World
    param waypoint: carla.Waypoint or carla.Transform
    param color: tuple, indicate color of poin and arrow
    """
    debug = world.debug

    if isinstance(waypoint, carla.Waypoint):
        transform = waypoint.transform
    elif isinstance(waypoint, carla.Transform):
        transform = waypoint
    else:
        raise ValueError('A waypoint or its transform is required for <draw_waypoint> method.')

    length = 1.0
    yaw = np.deg2rad(transform.rotation.yaw)
    vector = length * np.array([np.cos(yaw), np.sin(yaw)])

    start = transform.location
    end = start + carla.Location(x=vector[0], y=vector[1], z=0)

    # plot the waypoint
    # debug.draw_point(start, size=0.12, color=color[0], life_time=99999)
    debug.draw_arrow(start, end, thickness=0.15, arrow_size=0.15, color=color[1], life_time=99999)

    # print('waypoint is plot, tick the world to visualize it.')
    # world.tick()


def plot_actor_bbox(actor, thickness=0.5, color=red, duration_time=0.1):
    """
    Plot bounding box of a given actor.
    """
    # get carla.DebugHelper
    world = actor.get_world()
    debug_helper = world.debug

    transform = actor.get_transform()
    rotation = transform.rotation

    # todo check if actor has a bbox
    bbox = actor.bounding_box
    # relative location
    # todo which coord system is bbox location in???
    bbox.location = bbox.location + actor.get_location()

    debug_helper.draw_box(box=bbox,
                          rotation=rotation,
                          thickness=thickness,
                          color=color,
                          life_time=duration_time)

    # print('actor: ', actor.id, ' is plot in color: ', color)


def draw_2D_velocity(actor, length=1.0, color=magenta, life_time=1.0):
    """
    Draw 2D velocity(Vx, Vy) vector in the server.

    :param actor: usually carla.Vehicle or carla.Walker
    :param length: length of the arrow
    """
    # get carla.DebugHelper
    world = actor.get_world()
    debug_helper = world.debug

    # arrow begin as vehicle center
    begin = actor.get_location()
    velo_vector = actor.get_velocity()  # carla.Vector3D

    # height of the arrow
    height = 2.0
    begin.z = height

    # minimum speed
    speed_threshold = 0.1  # in m/s

    # need to transform to ndarray for normalization
    _velo_vector = np.array([velo_vector.x, velo_vector.y])
    speed_2D = np.linalg.norm(_velo_vector)

    # fix the arrow attribute when speed is low
    if speed_2D <= speed_threshold:
        length = 0.1
        color = red

    _velo_vector = length * _velo_vector / speed_2D

    # end point of velocity vector
    velo_vector = carla.Vector3D(_velo_vector[0], _velo_vector[1], 0.)  # relative vector, only x, y is considered,
    end = begin + velo_vector  # carla.Vector3D.__add__()

    debug_helper.draw_arrow(begin,  # carla.Location
                            end,
                            thickness=0.25,
                            arrow_size=0.1,
                            color=color,
                            life_time=life_time)


def get_azimuth(view_vector):
    """
        Get current azimuth from direction vector.
        Using only yaw and pitch angle.
    :param view_vector: vector of view direction in vehicle coord frame
    :return: azimuth in degrees, carla.Rotation
    """

    # transform matrix from World coord to ego vehicle
    # only yaw rotation is considered
    # np.cos(np.radians())
    # M1 = np.array([np.cos])

    # rotation around y axis
    x = view_vector[0]
    y = view_vector[1]
    z = view_vector[2]
    yaw = np.rad2deg(np.arctan2(y, x))
    pitch = np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y)))

    roll = 0  # always not using roll
    azimuth = {"pitch": pitch, "yaw": yaw, "roll": roll}
    # rotation = carla.Rotation(yaw=yaw, roll=0, pitch=pitch)
    return azimuth


def get_rotation_matrix_2D(transform):
    """
        Get a 2D transform matrix of a specified transform
        from actor reference frame to map coordinate frame
    :param transform: actor transform, actually only use yaw angle
    :return: rotation matrix
    """
    yaw = np.deg2rad(transform.rotation.yaw)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rotation_matrix_2D = np.array([[cy, -sy],
                                   [sy, cy]])
    return rotation_matrix_2D


def plot_local_coordinate_frame(world, origin_transform, axis_length_scale=3, life_time=99999, color_scheme=0):
    """
        Plot local coordinate frame.
        Using initial vehicle transform as origin
        todo: add text to identify axis, set x-axis always along longitudinal direction
    :param origin_transform: origin of local frame, in class transform
    :param axis_length_scale: length scale for axis vector
    :return: none, plot vectors of 3 axis in server world
    """

    # for test
    # origin_transform = transform
    # axis_length_scale = 3

    # longitudinal direction(x-axis)
    global x_axis_color
    yaw = np.deg2rad(origin_transform.rotation.yaw)
    # x axis
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # get coords in array and uplift with a height
    h = 3
    Origin_coord = np.array(
        [origin_transform.location.x, origin_transform.location.y, origin_transform.location.z + h])
    # elevate z coordinate
    Origin_location = carla.Location(Origin_coord[0], Origin_coord[1], Origin_coord[2])
    # x axis destination
    x_des_coord = Origin_coord + axis_length_scale * np.array([cy, sy, 0])
    x_des = carla.Location(x_des_coord[0], x_des_coord[1], x_des_coord[2])
    # y axis destination
    y_des_coord = Origin_coord + axis_length_scale * np.array([-sy, cy, 0])
    y_des = carla.Location(y_des_coord[0], y_des_coord[1], y_des_coord[2])
    # z axis destination
    z_des_coord = Origin_coord + axis_length_scale * np.array([0, 0, 1])
    z_des = carla.Location(z_des_coord[0], z_des_coord[1], z_des_coord[2])

    """
        color for each axis, carla.Color
        x-axis red:     (255, 0, 0)
        y-axis green:   (0, 255, 0)
        z-axis blue:    (0, 0, 255)
    """
    if color_scheme == 0:
        x_axis_color = carla.Color(r=255, g=0, b=0)
        y_axis_color = carla.Color(r=0, g=255, b=0)
        z_axis_color = carla.Color(r=0, g=0, b=255)
    elif color_scheme == 1:
        x_axis_color = carla.Color(r=252, g=157, b=154)
        y_axis_color = carla.Color(r=131, g=175, b=155)
        z_axis_color = carla.Color(r=96, g=143, b=159)

    # axis feature
    # thickness = 0.1f
    # arrow_size = 0.1f

    # begin, end, thickness=0.1f, arrow_size=0.1f, color=(255,0,0), life_time=-1.0f
    # draw x axis
    world.debug.draw_arrow(Origin_location, x_des, color=x_axis_color, life_time=life_time)
    # draw y axis
    world.debug.draw_arrow(Origin_location, y_des, color=y_axis_color, life_time=life_time)
    # draw z axis
    world.debug.draw_arrow(Origin_location, z_des, color=z_axis_color, life_time=life_time)

    # draw axis text next to arrow
    offset = 0.5
    x_text = carla.Location(x_des_coord[0] + offset, x_des_coord[1] + offset, x_des_coord[2] + offset)
    y_text = carla.Location(y_des_coord[0] + offset, y_des_coord[1] + offset, y_des_coord[2] + offset)
    z_text = carla.Location(z_des_coord[0] + offset, z_des_coord[1] + offset, z_des_coord[2] + offset)
    world.debug.draw_string(x_text, text='x', color=x_axis_color)
    world.debug.draw_string(y_text, text='y', color=x_axis_color)
    world.debug.draw_string(z_text, text='z', color=x_axis_color)
