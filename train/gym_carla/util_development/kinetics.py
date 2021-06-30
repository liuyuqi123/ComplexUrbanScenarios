"""
Some methods for kenetics.s
"""

import carla

import numpy as np
import math


def angle_reg(angle):
    """
    Regularize angle into certain bound.
    default range is [-pi, pi]
    """

    while True:
        if -np.pi <= angle <= np.pi:
            return angle

        if angle < -np.pi:
            angle += 2 * np.pi
        else:
            angle -= 2 * np.pi


def get_transform_matrix(transform: carla.Transform):
    """
    Get and parse a transformation matrix by transform.

    Matrix is from Actor coord system to the world coord system.

    :param transform:
    :return trans_matrix: transform matrix in ndarray
    """
    # original trans matrix in list
    _T = transform.get_matrix()

    # transform matrix from Actor system to world system
    trans_matrix = np.array([[_T[0][0], _T[0][1], _T[0][2]],
                             [_T[1][0], _T[1][1], _T[1][2]],
                             [_T[2][0], _T[2][1], _T[2][2]]])

    return trans_matrix


def get_inverse_transform_matrix(transform: carla.Transform):
    """
    Get inverse transform matrix from a transform class.

    Inverse transform refers to from world coord system to actor coord system.
    """
    _T = transform.get_inverse_matrix()

    # transform matrix from Actor system to world system
    inverse_trans_matrix = np.array([[_T[0][0], _T[0][1], _T[0][2]],
                                     [_T[1][0], _T[1][1], _T[1][2]],
                                     [_T[2][0], _T[2][1], _T[2][2]]])

    return inverse_trans_matrix


def vector2array(vector: carla.Vector3D):
    """
    Transform carla.Vector3D instance to ndarray
    """

    array = np.array([vector.x, vector.y, vector.z])

    return array


def get_vehicle_kinetic(vehicle: carla.Vehicle):
    """
    todo unfinished

    Get kinetics of ego vehicle.

    todo use a class to encapsulate all methods about getting kinetics
    """

    kinetic_dict = {}

    transform = vehicle.get_transform()

    vehicle.get_acceleration()
    vehicle.get_angular_velocity()


def get_distance_along_route(wmap, route, target_location):
    """
    Calculate the distance of the given location along the route

    Note: If the location is not along the route, the route length will be returned

    :param wmap: carla.Map of current world
    :param route: list of tuples, (carla.Transform, RoadOption)
    :param target_location:
    """

    covered_distance = 0
    prev_position = None
    found = False

    # Don't use the input location, use the corresponding wp as location
    target_location_from_wp = wmap.get_waypoint(target_location).transform.location

    for trans, _ in route:
        # input route is transform
        position = trans.location

        location = target_location_from_wp

        # Don't perform any calculations for the first route point
        if not prev_position:
            prev_position = position
            continue

        # Calculate distance between previous and current route point
        interval_length_squared = ((prev_position.x - position.x) ** 2) + ((prev_position.y - position.y) ** 2)
        distance_squared = ((location.x - prev_position.x) ** 2) + ((location.y - prev_position.y) ** 2)

        # Close to the current position? Stop calculation
        if distance_squared < 1.0:
            break

        if distance_squared < 400 and not distance_squared < interval_length_squared:
            # Check if a neighbor lane is closer to the route
            # Do this only in a close distance to correct route interval, otherwise the computation load is too high
            starting_wp = wmap.get_waypoint(location)
            wp = starting_wp.get_left_lane()
            while wp is not None:
                new_location = wp.transform.location
                new_distance_squared = ((new_location.x - prev_position.x) ** 2) + (
                    (new_location.y - prev_position.y) ** 2)

                if np.sign(starting_wp.lane_id) != np.sign(wp.lane_id):
                    break

                if new_distance_squared < distance_squared:
                    distance_squared = new_distance_squared
                    location = new_location
                else:
                    break

                wp = wp.get_left_lane()

            wp = starting_wp.get_right_lane()
            while wp is not None:
                new_location = wp.transform.location
                new_distance_squared = ((new_location.x - prev_position.x) ** 2) + (
                    (new_location.y - prev_position.y) ** 2)

                if np.sign(starting_wp.lane_id) != np.sign(wp.lane_id):
                    break

                if new_distance_squared < distance_squared:
                    distance_squared = new_distance_squared
                    location = new_location
                else:
                    break

                wp = wp.get_right_lane()

        if distance_squared < interval_length_squared:
            # The location could be inside the current route interval, if route/lane ids match
            # Note: This assumes a sufficiently small route interval
            # An alternative is to compare orientations, however, this also does not work for
            # long route intervals

            curr_wp = wmap.get_waypoint(position)
            prev_wp = wmap.get_waypoint(prev_position)
            wp = wmap.get_waypoint(location)

            if prev_wp and curr_wp and wp:
                if wp.road_id == prev_wp.road_id or wp.road_id == curr_wp.road_id:
                    # Roads match, now compare the sign of the lane ids
                    if (np.sign(wp.lane_id) == np.sign(prev_wp.lane_id) or
                            np.sign(wp.lane_id) == np.sign(curr_wp.lane_id)):
                        # The location is within the current route interval
                        covered_distance += math.sqrt(distance_squared)
                        found = True
                        break

        covered_distance += math.sqrt(interval_length_squared)
        prev_position = position

    return covered_distance, found


