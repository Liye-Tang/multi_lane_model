import numpy as np
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import math
import gym
from collections import OrderedDict


@dataclass
class Para:
    # reference info
    N = 10
    REF_RATIO = 8
    EXPECTED_V = 5
    INIT_UP_INDEX = 200

    # dim
    VEH_NUM = 6
    PER_VEH_INFO_DIM = 4
    EGO_DIM: int = 6
    GOAL_DIM: int = 3
    TRACK_DIM = 3
    CLOST_POINT_DIM = 3
    REF_DIM = N * 3
    VEH_DIM = PER_VEH_INFO_DIM * VEH_NUM

    # reward hparam
    scale_devi_p: float = 1
    scale_devi_v: float = 0.1
    scale_devi_phi: float = 0.8
    scale_punish_yaw_rate: float = 0  # 0.02
    scale_punish_steer: float = 0  # 0.2
    scale_punish_a_x: float = 0  # 0.02
    punish_factor = 10

    # action scale factor
    ACC_SCALE: float = 3.0
    ACC_SHIFT: float = 1.0
    STEER_SCALE: float = 0.3
    STEER_SHIFT: float = 0
    OBS_SCALE = [1/10, 1/10, 1/5, 1/100, 1/100, 1/180] + \
                [1/10, 1/90, 1/5] + \
                [1/100, 1/100, 1/180] + \
                [1/100, 1/100, 1/180, 1/5] * VEH_NUM + \
                [1/100, 1/100, 1/180] * N + \
                [1, 1, 1/4, 1/5]

    # done
    POS_TOLERANCE: float = 20
    ANGLE_TOLERANCE: float = 30.0

    # ego shape
    L: float = 4.8
    W: float = 2.0

    # goal
    GOAL_X_LOW: float = -40.
    GOAL_X_UP: float = 40.
    GOAL_Y_LOW: float = 40.
    GOAL_Y_UP: float = 60.
    GOAL_PHI_LOW: float = 0.
    GOAL_PHI_UP: float = 180.

    # ref path
    METER_POINT_NUM: int = 30
    START_LENGTH: float = 5.
    END_LENGTH: float = 5.

    # initial obs noise
    Y_RANGE = EXPECTED_V * 0.1
    # MU_PHI: float = 0
    # SIGMA_PHI: float = 20
    # MU_X: float = 0
    # SIGMA_X: float = 1
    # MU_Y: float = 0
    # SIGMA_Y: float = 1

    # simulation settings
    FREQUENCY: float = 10
    MAX_STEP = 200


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"))
        high = np.full(observation.shape, float("inf"))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def cal_eu_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def action_denormalize(action_norm):
    action = np.clip(action_norm, -1.05, 1.05)
    steer_norm, a_x_norm = action[0], action[1]
    scaled_steer = Para.STEER_SCALE * steer_norm - Para.STEER_SHIFT
    scaled_acc = Para.ACC_SCALE * a_x_norm - Para.ACC_SHIFT
    scaled_action = np.array([scaled_steer, scaled_acc], dtype=np.float32)
    return scaled_action


def process_obs(obs):
    return obs * Para.OBS_SCALE

def draw_rotate_rec(x, y, a, l, w):
    return matplotlib.patches.Rectangle((-l / 2 + x, -w / 2 + y),
                                        width=l, height=w,
                                        fill=False,
                                        facecolor=None,
                                        edgecolor='r',
                                        linewidth=1.0,
                                        transform=Affine2D().rotate_deg_around(*(x, y),
                                                                               a))


def draw_rotate_batch_rec(x, y, a, l, w):
    patch_list = []
    for i in range(len(x)):
        patch_list.append(matplotlib.patches.Rectangle(np.array([-l / 2 + x[i], -w / 2 + y[i]]),
                                     width=l, height=w,
                                     fill=False,
                                     facecolor=None,
                                     edgecolor='b',
                                     linewidth=1.0,
                                     transform=Affine2D().rotate_deg_around(*(x[i], y[i]),
                                                                               a[i])))
    return patch_list


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi


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
    transformed_d = np.where(transformed_d>180, transformed_d - 360, transformed_d)
    transformed_d = np.where(transformed_d<=-180, transformed_d + 360, transformed_d)
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


def ref_points_transform(untransformed_points):
    base_point = untransformed_points[:3, 0]
    base_point[2] -= 90
    transformed_pos = rotate_coordination_vec(untransformed_points[0] - base_point[0],
                                              untransformed_points[1] - base_point[1],
                                              untransformed_points[2],
                                              base_point[2])

    transformed_pos[2][0] = 90

    return np.stack((transformed_pos[0], transformed_pos[1], transformed_pos[2]))


def plot_multi_lane(ax, x, y, phi, left_lane, right_lane, lane_width):
    plot_lane_with_width(ax, x, y, phi, lane_width * 0.5)
    plot_lane_with_width(ax, x, y, phi, -lane_width * 0.5)
    if right_lane:
        plot_lane_with_width(ax, x, y, phi, lane_width * (0.5 + right_lane))
    if left_lane:
        plot_lane_with_width(ax, x, y, phi, -lane_width * (0.5 + left_lane))


def plot_lane_with_width(ax, x, y, phi, lane_width):
    theta = (phi - 90) / 180 * np.pi
    x_new = x + lane_width * np.cos(theta)
    y_new = y + lane_width * np.sin(theta)
    ax.plot(x_new, y_new, color='black')


def judge_point_line_pos(point, k, x_cl, y_cl):  # True for left
    x1, y1 = x_cl, y_cl
    x2, y2 = x_cl + 1, y_cl + k
    x3, y3, = point
    s = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    return s > 0


def cal_point_line_dis(point, k, x_cl, y_cl):
    ego_x, ego_y = point
    b = y_cl - k * x_cl
    x = (k * ego_y + ego_x - k * b) / (k ** 2 + 1)
    y = (k ** 2 * ego_y + k * ego_x + b) / (k ** 2 + 1)
    return np.square(ego_x - x) + np.square(ego_y - y)


def split_all(obs):
    obs_ego = obs[:Para.EGO_DIM]
    obs_track = obs[Para.EGO_DIM: Para.EGO_DIM + Para.TRACK_DIM]
    obs_closest_point = obs[(Para.EGO_DIM + Para.TRACK_DIM):
                                   (Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM)]
    obs_veh = obs[(Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM):
                  (Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.VEH_DIM)]
    obs_ref = obs[(Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.VEH_DIM):
                  (Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.VEH_DIM + Para.REF_DIM)]
    obs_left_lane = obs[(Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM):
                        (Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM + 1)]
    obs_right_lane = obs[(Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM + 1):
                         (Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM + 2)]
    obs_lane_width = obs[(Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM + 2):
                         (Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM + 3)]
    obs_ref_v = obs[(Para.EGO_DIM + Para.TRACK_DIM + Para.CLOST_POINT_DIM + Para.REF_DIM + Para.VEH_DIM + 3):]
    return obs_ego, obs_track, obs_closest_point, obs_veh, obs_ref, \
           obs_left_lane, obs_right_lane, obs_lane_width, obs_ref_v


def convert_to_rela(obs):
    obs_ego, obs_track, obs_closest_point, obs_veh, obs_ref, obs_left_lane, obs_right_lane, obs_lane_width, obs_ref_v = \
        split_all(obs)
    obs_veh_reshape = reshape_other(obs_veh)
    ego_x, ego_y = obs_ego[3], obs_ego[4]
    ego = np.array(([ego_x, ego_y] + [0.] * (Para.PER_VEH_INFO_DIM - 2)), dtype=np.float32)
    ego = ego[np.newaxis, :]
    rela = obs_veh_reshape - ego
    rela_obs_veh = reshape_other(rela, reverse=True)
    return np.concatenate(
        [obs_ego, obs_track, obs_closest_point, rela_obs_veh, obs_ref, obs_left_lane, obs_right_lane, obs_lane_width, obs_ref_v],
        axis=-1)


def _convert_to_abso(rela_obs):
    obs_ego, obs_track, obs_closest_point, obs_veh, obs_ref, obs_left_lane, obs_right_lane, obs_lane_width, obs_ref_v = \
        split_all(rela_obs)
    obs_veh_reshape = reshape_other(obs_veh)
    ego_x, ego_y = obs_ego[3], obs_ego[4]
    ego = np.array(([ego_x, ego_y] + [0.] * (Para.PER_VEH_INFO_DIM - 2)), dtype=np.float32)
    ego = ego[np.newaxis, :]
    abso = obs_veh_reshape + ego
    abso_obs_veh = reshape_other(abso, reverse=True)
    return np.concat(
        [obs_ego, obs_track, obs_closest_point, abso_obs_veh, obs_ref, obs_left_lane, obs_right_lane, obs_lane_width, obs_ref_v],
        axis=-1)


def reshape_other(obs_other, reverse=False):
    if reverse:
        return np.reshape(obs_other, (Para.VEH_NUM * Para.PER_VEH_INFO_DIM, ))
    else:
        return np.reshape(obs_other, (Para.VEH_NUM, Para.PER_VEH_INFO_DIM))


if __name__ == '__main__':
    a = Para.scale_devi_p * 0.25 + \
        Para.scale_devi_v * 25 + \
        Para.scale_devi_phi * np.square(30 / 180 * np.pi)
    print(a)
