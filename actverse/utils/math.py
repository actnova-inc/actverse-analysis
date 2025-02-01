import numpy as np


def vector_to_degree(vector):
    return np.degrees(np.arctan2(vector[1], vector[0]))


def correct_angle(current_angle, previous_angle):
    while abs(current_angle - previous_angle) > 180:
        if current_angle - previous_angle > 180:  # np.pi
            current_angle -= 360  # 2 * np.pi
        else:  # -np.pi
            current_angle += 360  # * np.pi
    return current_angle


def moving_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")
