#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend_env_utils.py
# =====================================

import math
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge


class ActionStore(list):
    def __init__(self, maxlen=2):
        super(ActionStore, self).__init__()
        self.maxlen = maxlen
        for _ in range(maxlen):
            self.append(np.zeros(2, dtype=np.float32))

    def put(self, action):
        assert len(self) == self.maxlen
        self.pop(0)
        self.append(action)

    def reset(self):
        for i in range(self.maxlen):
            self[i] = np.zeros(2, dtype=np.float32)


class Para:
    # MAP
    L, W = 4.8, 2.0
    LANE_WIDTH = 3.75
    BIKE_LANE_WIDTH = 0.0
    PERSON_LANE_WIDTH = 0.0
    LANE_NUMBER = 3
    CROSSROAD_SIZE = 50

    # DIM
    EGO_ENCODING_DIM = 6
    TRACK_ENCODING_DIM = 3
    LIGHT_ENCODING_DIM = 2
    TASK_ENCODING_DIM = 3
    REF_ENCODING_DIM = 3
    HIS_ACT_ENCODING_DIM = 4
    PER_OTHER_INFO_DIM = 10
    OTHER_START_DIM = EGO_ENCODING_DIM + TRACK_ENCODING_DIM + LIGHT_ENCODING_DIM + TASK_ENCODING_DIM + \
        REF_ENCODING_DIM + HIS_ACT_ENCODING_DIM

    # MAX NUM
    MAX_VEH_NUM = 6  # to be align with VEHICLE_MODE_DICT
    MAX_BIKE_NUM = 0  # to be align with BIKE_MODE_DICT
    MAX_PERSON_NUM = 0  # to be align with PERSON_MODE_DICT
    MAX_OTHER_NUM = MAX_VEH_NUM + MAX_BIKE_NUM + MAX_PERSON_NUM
    OBS_DIM = EGO_ENCODING_DIM + TRACK_ENCODING_DIM + LIGHT_ENCODING_DIM + TASK_ENCODING_DIM + \
              REF_ENCODING_DIM + HIS_ACT_ENCODING_DIM + MAX_OTHER_NUM * PER_OTHER_INFO_DIM
    FUTURE_POINT_NUM = 50
    # PATH_COLOR = ['blue', 'coral', 'darkcyan', 'pink']
    PATH_COLOR = ['k', 'k', 'k', 'k']

    STEER_SCALE = 0.3
    ACC_SHIFT, ACC_SCALE = -0.5, 1.5

    # NOISE
    # (v_x, v_y, r, x, y, phi) for ego
    # (x, y, v, phi, l, w; type encoding (d=3), turn rad) for other
    EGO_MEAN = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
    # EGO_VAR = np.diag([0.0418, 0.0418, 0., 0.0245, 0.0227, 0.0029*(180./np.pi)**2]).astype(np.float32)
    EGO_VAR = np.diag([0., 0., 0., 0., 0., 0.]).astype(np.float32)

    VEH_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_VEH_NUM)
    VEH_VAR = np.tile(np.array([0.0245, 0.0227, 0.0418, 0.0029 * (180. / np.pi) ** 2, 0.0902, 0.0202, 0., 0., 0., 0., ],
                               dtype=np.float32), MAX_VEH_NUM)

    BIKE_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_BIKE_NUM)
    BIKE_VAR = np.tile(np.array(
        [0.172 ** 2, 0.1583 ** 2, 0.1763 ** 2, (0.1707 * 180. / np.pi) ** 2, 0.1649 ** 2, 0.1091 ** 2, 0., 0., 0.,
         0., ], dtype=np.float32), MAX_BIKE_NUM)

    PERSON_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_PERSON_NUM)
    PERSON_VAR = np.tile(np.array(
        [0.1102 ** 2, 0.1108 ** 2, 0.1189 ** 2, (0.2289 * 180. / np.pi) ** 2, 0.1468 ** 2, 0.1405 ** 2, 0., 0., 0.,
         0., ], dtype=np.float32), MAX_PERSON_NUM)

    OTHERS_MEAN = np.concatenate([BIKE_MEAN, PERSON_MEAN, VEH_MEAN], axis=-1)  # order determined in line 735 in e2e.py
    OTHERS_VAR = np.diag(np.concatenate([BIKE_VAR, PERSON_VAR, VEH_VAR], axis=-1)).astype(np.float32)
    OBS_SCALE = [0.2, 1., 2., 1 / 30., 1 / 30, 1 / 180.] + \
                [1., 1 / 15., 0.2] + \
                [1., 1.] + \
                [1., 1., 1.] + \
                [1., 1., 1.] + \
                [1., 1., 1., 1.] + \
                [1 / 30., 1 / 30., 0.2, 1 / 180., 0.2, 0.5, 1., 1., 1., 0.] * MAX_OTHER_NUM


SUMOCFG_DIR = os.path.dirname(__file__) + "/sumo_files/cross.sumocfg"
VEHICLE_MODE_DICT = dict(left=OrderedDict(dl=2, du=2, ud=2, ul=2),
                         straight=OrderedDict(dl=1, du=2, ru=2, ur=2),
                         right=OrderedDict(dr=1, ur=2, lr=2))
BIKE_MODE_DICT = dict(left=OrderedDict(ud_b=2),
                      straight=OrderedDict(du_b=4),
                      right=OrderedDict(du_b=2, lr_b=0))
PERSON_MODE_DICT = dict(left=OrderedDict(c3=4),
                        straight=OrderedDict(c2=0),
                        right=OrderedDict(c1=4, c2=0))

# TODO(guanyang): check the correctness
LIGHT_PHASE_TO_GREEN_OR_RED = {0: 'green', 1: 'green', 2: 'red', 3: 'red',
                               4: 'red', 5: 'red'}  # 0: green, 1: red
TASK_ENCODING = dict(left=[1.0, 0.0, 0.0], straight=[0.0, 1.0, 0.0], right=[0.0, 0.0, 1.0])
LIGHT_ENCODING = {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0], 4: [0.0, 1.0], 5: [0.0, 1.0]}
REF_ENCODING = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0]}



def dict2flat(inp):
    out = []
    for key, val in inp.items():
        out.extend([key] * val)
    return out


def dict2num(inp):
    out = 0
    for _, val in inp.items():
        out += val
    return out


ROUTE2MODE = {('1o', '2i'): 'dr', ('1o', '3i'): 'du', ('1o', '4i'): 'dl',
              ('2o', '1i'): 'rd', ('2o', '3i'): 'ru', ('2o', '4i'): 'rl',
              ('3o', '1i'): 'ud', ('3o', '2i'): 'ur', ('3o', '4i'): 'ul',
              ('4o', '1i'): 'ld', ('4o', '2i'): 'lr', ('4o', '3i'): 'lu'}

MODE2TASK = {'dr': 'right', 'du': 'straight', 'dl': 'left',
             'rd': 'left', 'ru': 'right', 'rl': ' straight',
             'ud': 'straight', 'ur': 'left', 'ul': 'right',
             'ld': 'right', 'lr': 'straight', 'lu': 'left',
             'ud_b': 'straight', 'du_b': 'straight', 'lr_b': 'straight',
             'c1': 'straight', 'c2': 'straight', 'c3': 'straight'}

TASK2ROUTEID = {'left': 'dl', 'straight': 'du', 'right': 'dr'}

MODE2ROUTE = {'dr': ('1o', '2i'), 'du': ('1o', '3i'), 'dl': ('1o', '4i'),
              'rd': ('2o', '1i'), 'ru': ('2o', '3i'), 'rl': ('2o', '4i'),
              'ud': ('3o', '1i'), 'ur': ('3o', '2i'), 'ul': ('3o', '4i'),
              'ld': ('4o', '1i'), 'lr': ('4o', '2i'), 'lu': ('4o', '3i')}


def judge_feasible(orig_x, orig_y, task):  # map dependant
    def is_in_straight_before1(orig_x, orig_y):
        return 0 < orig_x < Para.LANE_WIDTH and orig_y <= -Para.CROSSROAD_SIZE / 2

    def is_in_straight_before2(orig_x, orig_y):
        return Para.LANE_WIDTH < orig_x < Para.LANE_WIDTH * 2 and \
               orig_y <= -Para.CROSSROAD_SIZE / 2

    def is_in_straight_before3(orig_x, orig_y):
        return Para.LANE_WIDTH * 2 < orig_x < Para.LANE_WIDTH * 3 and \
               orig_y <= -Para.CROSSROAD_SIZE / 2

    def is_in_straight_after(orig_x, orig_y):
        return 0 < orig_x < Para.LANE_WIDTH * Para.LANE_NUMBER and \
               orig_y >= Para.CROSSROAD_SIZE / 2

    def is_in_left(orig_x, orig_y):
        return 0 < orig_y < Para.LANE_WIDTH * Para.LANE_NUMBER and \
               orig_x < -Para.CROSSROAD_SIZE / 2

    def is_in_right(orig_x, orig_y):
        return -Para.LANE_WIDTH * Para.LANE_NUMBER < orig_y < 0 and \
               orig_x > Para.CROSSROAD_SIZE / 2

    def is_in_middle(orig_x, orig_y):
        return True if -Para.CROSSROAD_SIZE / 2 < orig_y < Para.CROSSROAD_SIZE / 2 and \
                       -Para.CROSSROAD_SIZE / 2 < orig_x < Para.CROSSROAD_SIZE / 2 else False

    if task == 'left':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False
    elif task == 'straight':
        return True if is_in_straight_before2(orig_x, orig_y) or is_in_straight_after(
            orig_x, orig_y) or is_in_middle(orig_x, orig_y) else False
    else:
        assert task == 'right'
        return True if is_in_straight_before3(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    """
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """
    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def rotate_coordination_vec(orig_x, orig_y, orig_d, coordi_rotate_d):
    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    transformed_d = np.where(transformed_d > 180, transformed_d - 360, transformed_d)
    transformed_d = np.where(transformed_d <= -180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def cal_info_in_transform_coordination(filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
    results = []
    for obj in filtered_objects:
        orig_x = obj['x']
        orig_y = obj['y']
        orig_v = obj['v']
        orig_heading = obj['phi']
        width = obj['w']
        length = obj['l']
        route = obj['route']
        shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
        trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
        trans_v = orig_v
        results.append({'x': trans_x,
                        'y': trans_y,
                        'v': trans_v,
                        'phi': trans_heading,
                        'w': width,
                        'l': length,
                        'route': route, })
    return results


def cal_ego_info_in_transform_coordination(ego_dynamics, x, y, rotate_d):
    orig_x, orig_y, orig_a, corner_points = ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'], ego_dynamics[
        'Corner_point']
    shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
    trans_x, trans_y, trans_a = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
    trans_corner_points = []
    for corner_x, corner_y in corner_points:
        shifted_x, shifted_y = shift_coordination(corner_x, corner_y, x, y)
        trans_corner_x, trans_corner_y, _ = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
        trans_corner_points.append((trans_corner_x, trans_corner_y))
    ego_dynamics.update(dict(x=trans_x,
                             y=trans_y,
                             phi=trans_a,
                             Corner_point=trans_corner_points))
    return ego_dynamics


def xy2_edgeID_lane(x, y):
    if y < -Para.CROSSROAD_SIZE / 2:
        edgeID = '1o'
        lane = int((Para.LANE_NUMBER - 1) - int(x / Para.LANE_WIDTH))
    elif x < -Para.CROSSROAD_SIZE / 2:
        edgeID = '4i'
        lane = int((Para.LANE_NUMBER - 1) - int(y / Para.LANE_WIDTH))
    elif y > Para.CROSSROAD_SIZE / 2:
        edgeID = '3i'
        lane = int((Para.LANE_NUMBER - 1) - int(x / Para.LANE_WIDTH))
    elif x > Para.CROSSROAD_SIZE / 2:
        edgeID = '2i'
        lane = int((Para.LANE_NUMBER - 1) - int(-y / Para.LANE_WIDTH))
    else:
        edgeID = '0'
        lane = 0
    return edgeID, lane


def convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi


def get_bezier_control_points(x1, y1, phi1, x4, y4, phi4):
    weight = 7 / 10
    x2 = x1 * ((np.cos(phi1) ** 2) * (1 - weight) + np.sin(phi1) ** 2) + \
         y1 * (-np.sin(phi1) * np.cos(phi1) * weight) + \
         x4 * ((np.cos(phi1) ** 2) * weight) + \
         y4 * (np.sin(phi1) * np.cos(phi1) * weight)
    y2 = x1 * (-np.sin(phi1) * np.cos(phi1) * weight) + \
         y1 * (np.cos(phi1) ** 2 + (np.sin(phi1) ** 2) * (1 - weight)) + \
         x4 * (np.sin(phi1) * np.cos(phi1) * weight) + \
         y4 * ((np.sin(phi1) ** 2) * weight)
    x3 = x1 * (np.cos(phi4) ** 2) * weight + \
         y1 * (np.sin(phi4) * np.cos(phi4) * weight) + \
         x4 * ((np.cos(phi4) ** 2) * (1 - weight) + np.sin(phi4) ** 2) + \
         y4 * (-np.sin(phi4) * np.cos(phi4) * weight)
    y3 = x1 * (np.sin(phi4) * np.cos(phi4) * weight) + \
         y1 * ((np.sin(phi4) ** 2) * weight) + \
         x4 * (-np.sin(phi4) * np.cos(phi4) * weight) + \
         y4 * (np.cos(phi4) ** 2 + (np.sin(phi4) ** 2) * (1 - weight))
    control_point2 = x2, y2
    control_point3 = x3, y3
    return control_point2, control_point3


def convert_to_rela(obs_abso):
    obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = split_all(obs_abso)
    obs_other_reshape = reshape_other(obs_other)
    ego_x, ego_y = obs_ego[3], obs_ego[4]
    ego = np.array(([ego_x, ego_y] + [0.] * (Para.PER_OTHER_INFO_DIM - 2)), dtype=np.float32)
    ego = ego[np.newaxis, :]
    rela = obs_other_reshape - ego
    rela_obs_other = reshape_other(rela, reverse=True)
    return np.concatenate([obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, rela_obs_other], axis=0)


def convert_to_abso(obs_rela):
    obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = split_all(obs_rela)
    obs_other_reshape = reshape_other(obs_other)
    ego_x, ego_y = obs_ego[3], obs_ego[4]
    ego = np.array(([ego_x, ego_y] + [0.] * (Para.PER_OTHER_INFO_DIM - 2)), dtype=np.float32)
    ego = ego[np.newaxis, :]
    abso = obs_other_reshape + ego
    abso_obs_other = reshape_other(abso, reverse=True)
    return np.concatenate([obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, abso_obs_other])


def split_all(obs):
    obs_ego = obs[:Para.EGO_ENCODING_DIM]
    obs_track = obs[Para.EGO_ENCODING_DIM:
                    Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM]
    obs_light = obs[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM:
                    Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.LIGHT_ENCODING_DIM]
    obs_task = obs[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.LIGHT_ENCODING_DIM:
                   Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM]
    obs_ref = obs[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM:
                  Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM +
                  Para.REF_ENCODING_DIM]
    obs_his_ac = obs[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM +
                     Para.REF_ENCODING_DIM:Para.OTHER_START_DIM]
    obs_other = obs[Para.OTHER_START_DIM:]

    return obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other


def split_other(obs_other):
    obs_bike = obs_other[:Para.MAX_BIKE_NUM * Para.PER_OTHER_INFO_DIM]
    obs_person = obs_other[Para.MAX_BIKE_NUM * Para.PER_OTHER_INFO_DIM:
                           (Para.MAX_BIKE_NUM + Para.MAX_PERSON_NUM) * Para.PER_OTHER_INFO_DIM]
    obs_veh = obs_other[(Para.MAX_BIKE_NUM + Para.MAX_PERSON_NUM) * Para.PER_OTHER_INFO_DIM:]
    return obs_bike, obs_person, obs_veh


def reshape_other(obs_other, reverse=False):
    if reverse:
        return np.reshape(obs_other, (Para.MAX_OTHER_NUM * Para.PER_OTHER_INFO_DIM,))
    else:
        return np.reshape(obs_other, (Para.MAX_OTHER_NUM, Para.PER_OTHER_INFO_DIM))


def render(light_phase, all_other, detected_other, interested_other, attn_weights, obs, ref_path,
           future_n_point, action, done_type, reward_info, hist_posi, path_values,
           sensor_config, is_debug):
    square_length = Para.CROSSROAD_SIZE
    extension = 40
    lane_width = Para.LANE_WIDTH
    lane_number = Para.LANE_NUMBER
    thin_linewidth = 0.3
    thick_linewidth = 1
    dotted_line_style = '--'
    solid_line_style = '-'

    plt.clf()
    ax = plt.axes([0.0, 0.0, 1, 1])
    ax.axis("off")
    ax.margins(0, 0)
    ax.axis("equal")
    patches = []

    # ----------horizon--------------
    ax.plot([-square_length / 2 - extension, -square_length / 2], [0.2, 0.2], color='orange')
    ax.plot([-square_length / 2 - extension, -square_length / 2], [-0.2, -0.2], color='orange')
    ax.plot([square_length / 2 + extension, square_length / 2], [0.2, 0.2], color='orange')
    ax.plot([square_length / 2 + extension, square_length / 2], [-0.2, -0.2], color='orange')

    for i in range(1, lane_number + 1):
        linestyle = dotted_line_style if i < lane_number else solid_line_style
        lw = thin_linewidth if i < lane_number else thick_linewidth
        ax.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                 linestyle=linestyle, color='black', linewidth=lw)
        ax.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                 linestyle=linestyle, color='black', linewidth=lw)
        ax.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                 linestyle=linestyle, color='black', linewidth=lw)
        ax.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                 linestyle=linestyle, color='black', linewidth=lw)

    # ----------vertical----------------
    ax.plot([0.2, 0.2], [-square_length / 2 - extension, -square_length / 2], color='orange')
    ax.plot([-0.2, -0.2], [-square_length / 2 - extension, -square_length / 2], color='orange')
    ax.plot([0.2, 0.2], [square_length / 2 + extension, square_length / 2], color='orange')
    ax.plot([-0.2, -0.2], [square_length / 2 + extension, square_length / 2], color='orange')

    for i in range(1, lane_number + 1):
        linestyle = dotted_line_style if i < lane_number else solid_line_style
        lw = thin_linewidth if i < lane_number else thick_linewidth
        ax.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=lw)
        ax.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=lw)
        ax.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=lw)
        ax.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=lw)

    # ----------stop line--------------
    if light_phase is not None:
        if light_phase == 0 or light_phase == 1:
            v_color, h_color = 'green', 'red'
        elif light_phase == 2:
            v_color, h_color = 'orange', 'red'
        elif light_phase == 3 or light_phase == 4:
            v_color, h_color = 'red', 'green'
        else:
            v_color, h_color = 'red', 'orange'
        ax.plot([0, (lane_number - 1) * lane_width], [-square_length / 2, -square_length / 2],
                 color=v_color, linewidth=thick_linewidth)
        ax.plot([(lane_number - 1) * lane_width, lane_number * lane_width], [-square_length / 2, -square_length / 2],
                 color='green', linewidth=thick_linewidth)

        ax.plot([-lane_number * lane_width, -(lane_number - 1) * lane_width], [square_length / 2, square_length / 2],
                 color='green', linewidth=thick_linewidth)
        ax.plot([-(lane_number - 1) * lane_width, 0], [square_length / 2, square_length / 2],
                 color=v_color, linewidth=thick_linewidth)

        ax.plot([-square_length / 2, -square_length / 2], [0, -(lane_number - 1) * lane_width],
                 color=h_color, linewidth=thick_linewidth)
        ax.plot([-square_length / 2, -square_length / 2], [-(lane_number - 1) * lane_width, -lane_number * lane_width],
                 color='green', linewidth=thick_linewidth)

        ax.plot([square_length / 2, square_length / 2], [(lane_number - 1) * lane_width, 0],
                 color=h_color, linewidth=thick_linewidth)
        ax.plot([square_length / 2, square_length / 2], [lane_number * lane_width, (lane_number - 1) * lane_width],
                 color='green', linewidth=thick_linewidth)
    else:
        ax.plot([0, lane_number * lane_width], [-square_length / 2, -square_length / 2], color='black', linewidth=thick_linewidth)
        ax.plot([-lane_number * lane_width, 0], [square_length / 2, square_length / 2], color='black', linewidth=thick_linewidth)
        ax.plot([-square_length / 2, -square_length / 2], [0, -lane_number * lane_width], color='black', linewidth=thick_linewidth)
        ax.plot([square_length / 2, square_length / 2], [lane_number * lane_width, 0], color='black', linewidth=thick_linewidth)

    # ----------Oblique--------------
    ax.plot([lane_number * lane_width, square_length / 2],
             [-square_length / 2, -lane_number * lane_width],
             color='black', linewidth=thick_linewidth)
    ax.plot([lane_number * lane_width, square_length / 2],
             [square_length / 2, lane_number * lane_width],
             color='black', linewidth=thick_linewidth)
    ax.plot([-lane_number * lane_width, -square_length / 2],
             [-square_length / 2, -lane_number * lane_width],
             color='black', linewidth=thick_linewidth)
    ax.plot([-lane_number * lane_width, -square_length / 2],
             [square_length / 2, lane_number * lane_width],
             color='black', linewidth=thick_linewidth)

    def is_in_plot_area(x, y, tolerance=5):
        if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
            return True
        else:
            return False

    def draw_rec(x, y, phi, l, w, facecolor, edgecolor, alpha=1.):
        phi_rad = phi * np.pi / 180
        bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -phi)
        ax.add_patch(plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=edgecolor, linewidth=thin_linewidth,
                                   facecolor=facecolor, angle=-(90 - phi), alpha=alpha, zorder=50))
        ax.plot([x, x+(l/2+1)*np.cos(phi_rad)], [y, y+(l/2+1)*np.sin(phi_rad)], linewidth=thin_linewidth, color=edgecolor)

    # plot others
    if all_other is not None:
        filted_all_other = [item for item in all_other if is_in_plot_area(item['x'], item['y'])]
        for item in filted_all_other:
            draw_rec(item['x'], item['y'], item['phi'], item['l'], item['w'], facecolor='white', edgecolor='k')

    # plot others
    if detected_other is not None:
        filted_all_other = [item for item in detected_other if is_in_plot_area(item['x'], item['y'])]
        for item in filted_all_other:
            draw_rec(item['x'], item['y'], item['phi'], item['l'], item['w'], facecolor='white', edgecolor='g')

    # plot attn weights
    if attn_weights is not None:
        assert attn_weights.shape == (Para.MAX_OTHER_NUM,), print(attn_weights.shape)
        index_top_k_in_weights = attn_weights.argsort()[-5:][::-1]
        for i, weight_idx in enumerate(index_top_k_in_weights):
            item = interested_other[i]
            weight = attn_weights[weight_idx]
            if item['y'] > -60 and is_in_plot_area(item['x'], item['y']):  # only plot real participants
                draw_rec(item['x'], item['y'], item['phi'], item['l'], item['w'],
                         facecolor='r', edgecolor='k', alpha=weight)
                # ax.text(item['x'], item['y'], "{:.2f}".format(attn_weights[i]), color='red', fontsize=15)

    # plot history
    if hist_posi is not None:
        freq = 5
        xs = [pos[0] for i, pos in enumerate(hist_posi) if i%freq==0 and is_in_plot_area(pos[0], pos[1], 1)]
        ys = [pos[1] for i, pos in enumerate(hist_posi) if i%freq==0 and is_in_plot_area(pos[0], pos[1], 1)]
        ts = [0.1*i for i, pos in enumerate(hist_posi) if i%freq==0 and is_in_plot_area(pos[0], pos[1], 1)]
        ax.scatter(np.array(xs), np.array(ys), marker='o', c=ts, cmap='Wistia', alpha=1, s=0.8, zorder=40)

    # plot ego
    if ref_path is not None:
        for i, path in enumerate(ref_path.path_list[LIGHT_PHASE_TO_GREEN_OR_RED[light_phase]]):
            if i == ref_path.path_index:
                ax.plot(path[0], path[1], color='k', alpha=1.0, linewidth=thin_linewidth)
            else:
                ax.plot(path[0], path[1], color='k', alpha=0.1, linewidth=thin_linewidth)

    if obs is not None:
        abso_obs = convert_to_abso(obs)
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = split_all(abso_obs)
        ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = obs_ego
        if is_in_plot_area(ego_x, ego_y):
            draw_rec(ego_x, ego_y, ego_phi, Para.L, Para.W, facecolor='fuchsia', edgecolor='k')
        # sensor config
        if sensor_config is not None:
            for prange, phirange in sensor_config:
                wedge = Wedge((ego_x, ego_y), prange, ego_phi-phirange/2, ego_phi+phirange/2, ec="none", alpha=0.1)
                patches.append(wedge)

    if is_debug:
        text_x, text_y_start = -110, 60
        ge = iter(range(0, 1000, 4))
        if interested_other is not None:
            filted_interested_other = [item for item in interested_other
                                       if is_in_plot_area(item['x'], item['y']) and item['exist']]
            for item in filted_interested_other:
                draw_rec(item['x'], item['y'], item['phi'], item['l'], item['w'],
                         facecolor='y', edgecolor='k')

        if obs is not None:
            abso_obs = convert_to_abso(obs)
            obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = split_all(abso_obs)
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = obs_ego
            devi_lateral, devi_phi, devi_v = obs_track
            ax.text(text_x, text_y_start - next(ge), '----ego info----')
            ax.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            ax.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            ax.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
            ax.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            ax.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            ax.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            ax.text(text_x, text_y_start - next(ge), '----track info----')
            ax.text(text_x, text_y_start - next(ge), 'devi_lateral: {:.2f}m'.format(devi_lateral))
            ax.text(text_x, text_y_start - next(ge), r'devi_phi: ${:.2f}\degree$'.format(devi_phi))
            ax.text(text_x, text_y_start - next(ge), 'devi_v: {:.2f}m/s'.format(devi_v))
            if ref_path is not None:
                _, point = ref_path._find_closest_point(ego_x, ego_y)
                path_x, path_y, path_phi, path_v = point[0], point[1], point[2], point[3]
                ax.plot(path_x, path_y, 'g.')
                ax.plot(future_n_point[0], future_n_point[1], 'g.')
                ax.text(text_x, text_y_start - next(ge), '----path info----')
                ax.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
                ax.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
        if path_values is not None:
            for i, value in enumerate(path_values):
                if i == ref_path.path_index:
                    ax.text(text_x, text_y_start - next(ge), 'path_cost={:.4f}'.format(value),
                            color='k', alpha=1.0)
                else:
                    ax.text(text_x, text_y_start - next(ge), 'path_cost={:.4f}'.format(value),
                            color='k', alpha=0.1)
        if action is not None:
            steer, a_x = action[0], action[1]
            ax.text(text_x, text_y_start - next(ge), '----act info----')
            ax.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            ax.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))
        text_x, text_y_start = 80, 60
        ge = iter(range(0, 1000, 4))
        # done info
        if done_type is not None:
            ax.text(text_x, text_y_start - next(ge), 'done info: {}'.format(done_type))
        # reward info
        if reward_info is not None:
            for key, val in reward_info.items():
                ax.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))
    ax.add_collection(PatchCollection(patches, match_original=True))
    return ax


if __name__ == '__main__':
    pass
