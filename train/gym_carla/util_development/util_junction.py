"""
Some methods for junction scenario.
"""

import carla

import time
import math
import numpy as np
import random


def get_next_junction(waypoint):
    """
    Get next junction along current waypoint route.

    :param waypoint: carla.Waypoint or carla.Location
    :return: carla.Junction
    """
    sampling_radius = 1.0
    while True:
        wp_choice = waypoint.next(sampling_radius)  # turning is ignored
        #   Choose path at intersection
        if len(wp_choice) > 1:
            # carla build-in critic
            reached_junction = wp_choice[0].is_junction
            if reached_junction:
                junction = wp_choice[0].get_junction()
                break
        else:
            waypoint = wp_choice[0]

    return junction


def plot_junction(junction):
    """
    todo unfinished
    Visualize a carla junction'.

    param junction: carla.Junction
    """
    bbox = junction.bounding_box

    # set spectator on the junction
    location = junction.bounding_box.location
    rotation = start_rotation
    transform = carla.Transform(location, rotation)
    # self.set_spectator(transform, 30)

    # ==================================================
    """
    # test carla.Junction get_waypoints method
    lane_type = wp_choice[0].lane_type
    wp_pair_list = junction.get_waypoints(lane_type)  # not quite understand how this list work
    # visualize wp_pair_list
    for tp in wp_pair_list:
        self.draw_waypoint(tp[0])
        self.draw_waypoint(tp[1])
    """
    # ==================================================

    # print('testing get_junction method')
    pass


def TransMatrix_yaw(transform):
    """
    Get a 2D transform matrix of yaw rotation
    :param transform: carla.transform, or yaw in degrees.
    :return: rotation matrix
    """
    if isinstance(transform, carla.Transform):
        yaw = np.deg2rad(transform.rotation.yaw)
    elif isinstance(transform, float):
        yaw = np.deg2rad(transform)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    tf_matrix = np.array([[cy, -sy],
                          [sy, cy]])
    return tf_matrix


def plot_coordinate_frame(debug, origin_transform, color=0, scale=1.0):
    """
    Plot local coordinate frame.
    Using initial vehicle transform as origin
    todo: draw string fails

    :param origin_transform: origin of local frame, in class transform
    :param axis_length_scale: length scale for axis vector
    :return: none, plot vectors of 3 axis in global coords
    """

    # for test
    # origin_transform = transform
    # axis_length_scale = 3

    # longitudinal direction(x-axis)
    yaw = np.deg2rad(origin_transform.rotation.yaw)
    # x axis
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # get coords in array and uplift a certain height
    h = 3
    Origin_coord = np.array(
        [origin_transform.location.x, origin_transform.location.y, h])
    Origin_location = carla.Location(Origin_coord[0], Origin_coord[1], Origin_coord[2])

    axis_length_scale = 3
    # x axis destination
    x_des_coord = Origin_coord + axis_length_scale * np.array([cy, sy, 0])
    x_des = carla.Location(x_des_coord[0], x_des_coord[1], x_des_coord[2])
    # y axis destination
    y_des_coord = Origin_coord + axis_length_scale * np.array([-sy, cy, 0])
    y_des = carla.Location(y_des_coord[0], y_des_coord[1], y_des_coord[2])
    # z axis destination
    z_des_coord = Origin_coord + axis_length_scale * np.array([0, 0, 1])
    z_des = carla.Location(z_des_coord[0], z_des_coord[1], z_des_coord[2])

    # set color scheme for axis
    if color == 0:
        x_axis_color = carla.Color(r=255, g=0, b=0)
        y_axis_color = carla.Color(r=0, g=255, b=0)
        z_axis_color = carla.Color(r=0, g=0, b=255)
    elif color == 1:
        x_axis_color = carla.Color(r=250, g=10, b=110)
        y_axis_color = carla.Color(r=10, g=250, b=10)
        z_axis_color = carla.Color(r=10, g=10, b=250)

    # axis feature
    life_time = 99999
    thickness = 0.1*scale
    arrow_size = 0.1*scale

    # begin, end, thickness=0.1f, arrow_size=0.1f, color=(255,0,0), life_time=-1.0f
    # draw 3 axis
    debug.draw_arrow(begin=Origin_location, end=x_des,
                           thickness=thickness,
                           arrow_size=arrow_size,
                           color=x_axis_color,
                           life_time=life_time)

    debug.draw_arrow(begin=Origin_location, end=y_des,
                           thickness=thickness,
                           arrow_size=arrow_size,
                           color=y_axis_color,
                           life_time=life_time)

    debug.draw_arrow(begin=Origin_location, end=z_des,
                           thickness=thickness,
                           arrow_size=arrow_size,
                           color=z_axis_color,
                           life_time=life_time)

    # draw axis text next to arrow
    offset = 0.5
    x_text_loc = carla.Location(x_des_coord[0]+offset, x_des_coord[1]+offset, x_des_coord[2]+offset)
    y_text_loc = carla.Location(y_des_coord[0]+offset, y_des_coord[1]+offset, y_des_coord[2]+offset)
    z_text_loc = carla.Location(z_des_coord[0]+offset, z_des_coord[1]+offset, z_des_coord[2]+offset)
    debug.draw_string(location=x_text_loc, text='x', color=x_axis_color)
    debug.draw_string(location=y_text_loc, text='y', color=y_axis_color)
    debug.draw_string(location=z_text_loc, text='z', color=z_axis_color)



