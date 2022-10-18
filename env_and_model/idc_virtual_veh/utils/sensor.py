#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/3/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: sensor.py.py
# =====================================
import copy
from env_and_model.idc_virtual_veh.endtoend_env_utils import *
import numpy as np


class Lidar:
    def __init__(self):
        self.prange = 80.
        self.noise_xy_std = 0.14
        self.noise_v_std = 0.15
        self.noise_phi_std = 1.
        self.noise_lw_std = 0.05

    def is_in_range(self, other_x_in_ego, other_y_in_ego, other_cen_direction):
        if np.sqrt(np.square(other_x_in_ego) + np.square(other_y_in_ego)) < self.prange:
            return True
        else:
            return False

    def process(self, other_x, other_y, other_phi, other_l, other_w):
        processed_other_x = other_x + np.random.normal(0, self.noise_xy_std)
        processed_other_y = other_y + np.random.normal(0, self.noise_xy_std)
        processed_other_phi = other_phi + np.random.normal(0, self.noise_phi_std)
        processed_other_l = other_l + np.random.normal(0, self.noise_lw_std)
        processed_other_w = other_w + np.random.normal(0, self.noise_lw_std)
        return processed_other_x, processed_other_y, processed_other_phi, \
               processed_other_l, processed_other_w


class Camera:
    def __init__(self,):
        self.prange = 100.
        self.phirange = 38. * np.pi / 180  # rad
        self.noise_xy_std = 0.48
        self.noise_v_std = 1.4
        self.noise_phi_std = 1.
        self.noise_lw_std = 0.06

    def is_in_range(self, other_x_in_ego, other_y_in_ego, other_cen_direction):
        if np.sqrt(np.square(other_x_in_ego) + np.square(other_y_in_ego)) < self.prange and \
                -self.phirange / 2 < other_cen_direction < self.phirange / 2:
            return True
        else:
            return False

    def process(self, other_x, other_y, other_phi, other_l, other_w):
        processed_other_x = other_x + np.random.normal(0, self.noise_xy_std)
        processed_other_y = other_y + np.random.normal(0, self.noise_xy_std)
        processed_other_phi = other_phi + np.random.normal(0, self.noise_phi_std)
        processed_other_l = other_l + np.random.normal(0, self.noise_lw_std)
        processed_other_w = other_w + np.random.normal(0, self.noise_lw_std)
        return processed_other_x, processed_other_y, processed_other_phi, \
               processed_other_l, processed_other_w


class Perception:
    def __init__(self,):
        self.ego_x, self.ego_y, self.ego_phi = None, None, None
        self.lidar = Lidar()
        self.camera = Camera()
        self.all_other = None

    def get_ego(self, ego_x, ego_y, ego_phi):
        self.ego_x, self.ego_y, self.ego_phi = ego_x, ego_y, ego_phi

    def transform2egocoordi(self, other_x, other_y, other_phi):
        x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = \
            shift_and_rotate_coordination(other_x, other_y, other_phi, self.ego_x, self.ego_y, self.ego_phi)
        return x_in_ego_coord, y_in_ego_coord, a_in_ego_coord

    def transform2worldcoordi(self, x_in_ego_coord, y_in_ego_coord, phi_in_ego_coord):
        x, y, phi = \
            rotate_and_shift_coordination(x_in_ego_coord, y_in_ego_coord, phi_in_ego_coord, -self.ego_x, -self.ego_y, -self.ego_phi)
        return x, y, phi

    def compute_directions(self, x, y, phi, l, w):  # phi is in deg
        x1, y1, _ = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
        x2, y2, _ = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
        x3, y3, _ = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
        x4, y4, _ = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
        cen_direction = np.arctan2(y, x)
        cor1_direction = np.arctan2(y1, x1)
        cor2_direction = np.arctan2(y2, x2)
        cor3_direction = np.arctan2(y3, x3)
        cor4_direction = np.arctan2(y4, x4)
        min_cor_direction, max_cor_direction = min(cor1_direction, cor2_direction, cor3_direction, cor4_direction), \
                                               max(cor1_direction, cor2_direction, cor3_direction, cor4_direction)

        if max_cor_direction - min_cor_direction > np.pi and min_cor_direction*max_cor_direction<0:
            other_ranges = [(-180., min_cor_direction), (max_cor_direction, 180.)]
        else:
            other_ranges = [(min_cor_direction, max_cor_direction)]

        return cen_direction, cor1_direction, cor2_direction, cor3_direction, cor4_direction, \
               other_ranges

    def process(self, all_other):
        # dict(type=type, x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route, road=road)
        self.all_other = copy.deepcopy(all_other)
        for other in self.all_other:
            other['rela_x'], other['rela_y'], other['rela_phi'] = \
                self.transform2egocoordi(other['x'], other['y'], other['phi'])
            other['distance'] = np.sqrt(np.square(other['rela_x']) + np.square(other['rela_y']))
            other['cen_direction'], other['cor1_direction'], other['cor2_direction'], \
            other['cor3_direction'], other['cor4_direction'], \
            other['ranges'] = \
                self.compute_directions(other['rela_x'], other['rela_y'], other['rela_phi'], other['l'], other['w'])
            # compute direction of center and corners
            if self.lidar.is_in_range(other['rela_x'], other['rela_y'], other['cen_direction']):
                other['is_detected'] = True
                other['rela_x'], other['rela_y'], other['rela_phi'], other['l'], other['w'] = \
                    self.lidar.process(other['rela_x'], other['rela_y'], other['rela_phi'], other['l'], other['w'])
            elif self.camera.is_in_range(other['rela_x'], other['rela_y'], other['cen_direction']):
                other['is_detected'] = True
                other['rela_x'], other['rela_y'], other['rela_phi'], other['l'], other['w'] = \
                    self.camera.process(other['rela_x'], other['rela_y'], other['rela_phi'], other['l'], other['w'])
            else:
                other['is_detected'] = False
        self.all_other = [other for other in self.all_other if other['is_detected']]

        # deal with shelter
        shelter_ranges = []
        self.all_other.sort(key=lambda v: v['distance'])
        for other in self.all_other:
            is_cor1_shelted = sum([low < other['cor1_direction'] < high for low, high in shelter_ranges]) > 0
            is_cor2_shelted = sum([low < other['cor2_direction'] < high for low, high in shelter_ranges]) > 0
            is_cor3_shelted = sum([low < other['cor3_direction'] < high for low, high in shelter_ranges]) > 0
            is_cor4_shelted = sum([low < other['cor4_direction'] < high for low, high in shelter_ranges]) > 0
            if sum([is_cor1_shelted, is_cor2_shelted, is_cor3_shelted, is_cor4_shelted]) > 1:
                other['is_detected'] = False
            # is_cen_shelted = sum([low < other['cen_direction'] < high for low, high in shelter_ranges]) > 0
            # if is_cen_shelted:
            #     other['is_detected'] = False
            shelter_ranges.extend(other['ranges'])
        self.all_other = [other for other in self.all_other if other['is_detected']]

        for other in self.all_other:
            other['x'], other['y'], other['phi'] = \
                self.transform2worldcoordi(other['rela_x'], other['rela_y'], other['rela_phi'])
        return self.all_other


def test11(a):
    if a < 5:
        print(1)
    elif a < 6:
        print(2)
    else:
        print(3)


def test_perception():
    all_other = [{'x': -1.875, 'y': 24., 'phi': -60., 'l': 5., 'w':2.5}]
    perc = Perception()
    perc.get_ego(1.875, -25., 90.)
    print(perc.process(all_other))


if __name__ == "__main__":
    test_perception()