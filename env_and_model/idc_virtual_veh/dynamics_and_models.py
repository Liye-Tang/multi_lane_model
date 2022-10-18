#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from math import pi

import bezier
import tensorflow as tf
from tensorflow import logical_and

from env_and_model.idc_virtual_veh.endtoend_env_utils import *

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class VehicleDynamics(object):
    def __init__(self, if_model):
        # TODO(guanyang): determine these parameters
        # self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
        #                            C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
        #                            a=1.286,  # distance from CG to front axle [m]
        #                            b=1.578,  # distance from CG to rear axle [m]
        #                            mass=1520.,  # mass [kg]
        #                            I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
        #                            miu=0.8,  # tire-road friction coefficient
        #                            g=9.81,  # acceleration of gravity [m/s^2]
        #                            )
        self.vehicle_params = dict(C_f=-128916.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85944.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        # TODO: use it in the f_xu
        self.if_model = if_model

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x + 1e-8)) - steer
        alpha_r = tf.atan((v_y - b * r) / (v_x + 1e-8))

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


# class VehicleDynamics(object):
#     def __init__(self, ):
#         self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
#                                    C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
#                                    a=1.19,  # distance from CG to front axle [m]
#                                    b=1.46,  # distance from CG to rear axle [m]
#                                    mass=1520.,  # mass [kg]
#                                    I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
#                                    miu=0.8,  # tire-road friction coefficient
#                                    g=9.81,  # acceleration of gravity [m/s^2]
#                                    )
#         a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
#                         self.vehicle_params['mass'], self.vehicle_params['g']
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         self.vehicle_params.update(dict(F_zf=F_zf,
#                                         F_zr=F_zr))
#
#     def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
#         v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
#         phi = phi * np.pi / 180.
#         steer, a_x = actions[:, 0], actions[:, 1]
#         next_v_x = v_x + tau * a_x
#         next_x = x + tau * v_x * tf.cos(phi)
#         next_y = y + tau * v_x * tf.sin(phi)
#         next_phi = phi + tau * v_x * tf.tan(steer) / 2.865
#         next_phi_deg = next_phi * 180 / np.pi
#         next_r = v_x * tf.tan(steer) / 2.865
#         next_v_y = tf.zeros_like(next_v_x)
#         next_state = [next_v_x, next_v_y, next_r, next_x, next_y, next_phi_deg]
#         return tf.stack(next_state, 1), tf.zeros_like(states[:, :4])
#
#     def prediction(self, x_1, u_1, frequency):
#         x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
#         return x_next, next_params


class IdcVirtualVehModel(object):  # all tensors
    def __init__(self, **kwargs):
        self.if_model = False if kwargs.get('use_model_as_env') else True
        self.vehicle_dynamics = VehicleDynamics(if_model=self.if_model)
        self.base_frequency = 10.
        self.obses = None
        self.actions = None
        self.reward_dict = None
        self.future_n_points = None
        self.path_len = None
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.ref_info_dim = Para.REF_ENCODING_DIM
        self.his_act_info_dim = Para.HIS_ACT_ENCODING_DIM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.light_info_dim,
                                    self.task_info_dim, self.ref_info_dim, self.his_act_info_dim])
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM

    def reset(self, obses, future_n_points):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_dict = None
        self.future_n_points = future_n_points
        self.path_len = self.future_n_points.shape[-1]

    def rollout(self, actions):  # ref_points [#batch, 4]
        self.actions = self._action_transformation_for_end2end(actions)
        self.reward_dict = self.compute_rewards(self.obses, self.actions, actions)
        self.obses = self.compute_next_obses(self.obses, self.actions, actions)
        return self.obses, self.reward_dict

    def rollout_online(self, actions):  # ref_points [#batch, 4]
        self.actions = self._action_transformation_for_end2end(actions)
        reward_dict = self.compute_rewards(self.obses, self.actions, actions)
        self.obses = self.compute_next_obses(self.obses, self.actions, actions)
        return self.obses, reward_dict

    def compute_rewards(self, obses, actions, untransformed_action):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)
        obses_bike, obses_person, obses_veh = self._split_other(obses_other)

        with tf.name_scope('compute_reward') as scope:
            bike_infos = tf.stop_gradient(obses_bike)
            person_infos = tf.stop_gradient(obses_person)
            veh_infos = tf.stop_gradient(obses_veh)
            obses_his_ac = tf.stop_gradient(obses_his_ac)

            steers, a_xs = actions[:, 0], actions[:, 1]
            steers_t_minus_2, a_x_t_minus_2 = obses_his_ac[:, 0], obses_his_ac[:, 1]
            steers_t_minus_1, a_x_t_minus_1 = obses_his_ac[:, 2], obses_his_ac[:, 3]
            steers_t, a_x_t = untransformed_action[:, 0], untransformed_action[:, 1]

            # rewards related to action and its derivatives
            steers_1st_order = (steers_t - steers_t_minus_1) * self.base_frequency
            steers_2nd_order = (steers_t - 2 * steers_t_minus_1 + steers_t_minus_2) * (
                    self.base_frequency ** 2)
            a_xs_1st_order = (a_x_t - a_x_t_minus_1) * self.base_frequency
            a_xs_2nd_order = (a_x_t - 2 * a_x_t_minus_1 + a_x_t_minus_2) * (
                    self.base_frequency ** 2)

            rew_steer0, rew_steer1, rew_steer2 = \
                tf.square(steers), tf.square(steers_1st_order), tf.square(steers_2nd_order)
            rew_a_x0, rew_a_x1, rew_a_x2 = \
                tf.square(a_xs), tf.square(a_xs_1st_order), tf.square(a_xs_2nd_order)

            # rewards related to ego stability
            rew_vy = tf.square(obses_ego[:, 1])
            rew_yaw_rate = tf.square(obses_ego[:, 2])

            # rewards related to tracking error
            rew_devi_lateral = tf.square(obses_track[:, 0])
            rew_devi_phi = tf.cast(tf.square(obses_track[:, 1] * np.pi / 180.), dtype=tf.float32)
            rew_devi_v = tf.square(obses_track[:, 2])

            # rewards related to veh2veh collision
            ego_lws = (Para.L - Para.W) / 2.
            ego_front_points = tf.cast(obses_ego[:, 3] + ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32), \
                               tf.cast(obses_ego[:, 4] + ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32)
            ego_rear_points = tf.cast(obses_ego[:, 3] - ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32), \
                              tf.cast(obses_ego[:, 4] - ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32)

            delta_ego_vs = obses_ego[:, 0] - 8.33  # delta_ego_vs = ego_vs - ref_vs
            punish_veh2speed4training = tf.where(delta_ego_vs > 0.0, tf.square(delta_ego_vs), tf.zeros_like(veh_infos[:, 0]))
            punish_veh2speed4real = tf.where(delta_ego_vs > 0.0, tf.square(delta_ego_vs), tf.zeros_like(veh_infos[:, 0]))

            punish_veh2veh4real = tf.zeros_like(veh_infos[:, 0])
            punish_veh2veh4training = tf.zeros_like(veh_infos[:, 0])

            for veh_index in range(self.veh_num):
                vehs = veh_infos[:, veh_index * self.per_other_info_dim:(veh_index + 1) * self.per_other_info_dim]
                veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(
                            tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
                        punish_veh2veh4training += tf.where(veh2veh_dist - 3.5 < 0, tf.square(veh2veh_dist - 3.5),
                                                            tf.zeros_like(veh_infos[:, 0]))
                        punish_veh2veh4real += tf.where(veh2veh_dist - 2.5 < 0, tf.square(veh2veh_dist - 2.5),
                                                        tf.zeros_like(veh_infos[:, 0]))

            punish_veh2bike4real = tf.zeros_like(veh_infos[:, 0])
            punish_veh2bike4training = tf.zeros_like(veh_infos[:, 0])
            for bike_index in range(self.bike_num):
                bikes = bike_infos[:, bike_index * self.per_other_info_dim:(bike_index + 1) * self.per_other_info_dim]
                bike_lws = (bikes[:, 4] - bikes[:, 5]) / 2.
                bike_front_points = tf.cast(bikes[:, 0] + bike_lws * tf.cos(bikes[:, 3] * np.pi / 180.),
                                            dtype=tf.float32), \
                                    tf.cast(bikes[:, 1] + bike_lws * tf.sin(bikes[:, 3] * np.pi / 180.),
                                            dtype=tf.float32)
                bike_rear_points = tf.cast(bikes[:, 0] - bike_lws * tf.cos(bikes[:, 3] * np.pi / 180.),
                                           dtype=tf.float32), \
                                   tf.cast(bikes[:, 1] - bike_lws * tf.sin(bikes[:, 3] * np.pi / 180.),
                                           dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for bike_point in [bike_front_points, bike_rear_points]:
                        veh2bike_dist = tf.sqrt(
                            tf.square(ego_point[0] - bike_point[0]) + tf.square(ego_point[1] - bike_point[1]))
                        punish_veh2bike4training += tf.where(veh2bike_dist - 3.5 < 0, tf.square(veh2bike_dist - 3.5),
                                                             tf.zeros_like(veh_infos[:, 0]))
                        punish_veh2bike4real += tf.where(veh2bike_dist - 2.5 < 0, tf.square(veh2bike_dist - 2.5),
                                                         tf.zeros_like(veh_infos[:, 0]))

            punish_veh2person4real = tf.zeros_like(veh_infos[:, 0])
            punish_veh2person4training = tf.zeros_like(veh_infos[:, 0])
            for person_index in range(self.person_num):
                persons = person_infos[:,
                          person_index * self.per_other_info_dim:(person_index + 1) * self.per_other_info_dim]
                person_point = tf.cast(persons[:, 0], dtype=tf.float32), tf.cast(persons[:, 1], dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    veh2person_dist = tf.sqrt(
                        tf.square(ego_point[0] - person_point[0]) + tf.square(ego_point[1] - person_point[1]))
                    punish_veh2person4training += tf.where(veh2person_dist - 3.5 < 0, tf.square(veh2person_dist - 3.5),
                                                           tf.zeros_like(veh_infos[:, 0]))
                    punish_veh2person4real += tf.where(veh2person_dist - 2.5 < 0, tf.square(veh2person_dist - 2.5),
                                                       tf.zeros_like(veh_infos[:, 0]))

            punish_veh2road4real = tf.zeros_like(veh_infos[:, 0])
            punish_veh2road4training = tf.zeros_like(veh_infos[:, 0])
            for ego_point in [ego_front_points, ego_rear_points]:
                left_flag = tf.reduce_all(tf.math.equal(obses_task, [[1., 0., 0.]]), axis=1, keepdims=False)
                straight_flag = tf.reduce_all(tf.math.equal(obses_task, [[0., 1., 0.]]), axis=1, keepdims=False)
                right_flag = tf.reduce_all(tf.math.equal(obses_task, [[0., 0., 1.]]), axis=1, keepdims=False)
                # punish_veh2road4training += tf.where(
                #     logical_and(left_flag, logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2, ego_point[0] < 1)),
                #     tf.square(ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                # punish_veh2road4training += tf.where(logical_and(left_flag,
                #                                           logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2,
                #                                                       Para.LANE_WIDTH - ego_point[0] < 1)),
                #                               tf.square(Para.LANE_WIDTH - ego_point[0] - 1),
                #                               tf.zeros_like(veh_infos[:, 0]))
                punish_veh2road4training += tf.where(logical_and(left_flag, logical_and(ego_point[0] < 0,
                                                                                 Para.LANE_WIDTH * Para.LANE_NUMBER -
                                                                                 ego_point[1] < 1)),
                                              tf.square(Para.LANE_WIDTH * Para.LANE_NUMBER - ego_point[1] - 1),
                                              tf.zeros_like(veh_infos[:, 0]))
                punish_veh2road4training += tf.where(
                    logical_and(left_flag, logical_and(ego_point[0] < -Para.CROSSROAD_SIZE / 2, ego_point[1] - 0 < 1)),
                    tf.square(ego_point[1] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))

                # punish_veh2road4training += tf.where(logical_and(straight_flag,
                #                                           logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2,
                #                                                       ego_point[0] - Para.LANE_WIDTH < 1)),
                #                               tf.square(ego_point[0] - Para.LANE_WIDTH - 1),
                #                               tf.zeros_like(veh_infos[:, 0]))
                # punish_veh2road4training += tf.where(logical_and(straight_flag,
                #                                           logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2,
                #                                                       2 * Para.LANE_WIDTH - ego_point[0] < 1)),
                #                               tf.square(2 * Para.LANE_WIDTH - ego_point[0] - 1),
                #                               tf.zeros_like(veh_infos[:, 0]))
                punish_veh2road4training += tf.where(logical_and(straight_flag,
                                                          logical_and(ego_point[1] > Para.CROSSROAD_SIZE / 2,
                                                                      Para.LANE_WIDTH * Para.LANE_NUMBER - ego_point[
                                                                          0] < 1)),
                                              tf.square(Para.LANE_WIDTH * Para.LANE_NUMBER - ego_point[0] - 1),
                                              tf.zeros_like(veh_infos[:, 0]))
                punish_veh2road4training += tf.where(logical_and(straight_flag,
                                                          logical_and(ego_point[1] > Para.CROSSROAD_SIZE / 2,
                                                                      ego_point[0] - 0 < 1)),
                                              tf.square(ego_point[0] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))

                # punish_veh2road4training += tf.where(logical_and(right_flag,
                #                                           logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2,
                #                                                       ego_point[0] - 2 * Para.LANE_WIDTH < 1)),
                #                               tf.square(ego_point[0] - 2 * Para.LANE_WIDTH - 1),
                #                               tf.zeros_like(veh_infos[:, 0]))
                # punish_veh2road4training += tf.where(logical_and(right_flag,
                #                                           logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2,
                #                                                       Para.LANE_NUMBER * Para.LANE_WIDTH - ego_point[
                #                                                           0] < 1)),
                #                               tf.square(Para.LANE_NUMBER * Para.LANE_WIDTH - ego_point[0] - 1),
                #                               tf.zeros_like(veh_infos[:, 0]))
                punish_veh2road4training += tf.where(
                    logical_and(right_flag, logical_and(ego_point[0] > Para.CROSSROAD_SIZE / 2, 0 - ego_point[1] < 1)),
                    tf.square(0 - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
                punish_veh2road4training += tf.where(logical_and(right_flag,
                                                                 logical_and(ego_point[0] > Para.CROSSROAD_SIZE / 2,
                                                                             ego_point[1] - (-Para.LANE_WIDTH * Para.LANE_NUMBER) < 1)),
                                                     tf.square(ego_point[1] - (-Para.LANE_WIDTH * Para.LANE_NUMBER) - 1),
                                                     tf.zeros_like(veh_infos[:, 0]))

            punish_veh2road4real = punish_veh2road4training

            scale = dict(rew_devi_v=-0.4, rew_devi_lateral=-0.6, rew_devi_phi=-20., rew_yaw_rate=-0.02, rew_vy=-0.01,
                         rew_steer0=-5., rew_a_x0=-0.05, rew_steer1=-0.04, rew_a_x1=-0.01,
                         rew_devi_v_4value=-0.15, rew_devi_lateral_4value=-0.4, rew_devi_phi_4value=-20.,
                         rew_yaw_rate_4value=-0.02, rew_vy_4value=-0.01, rew_steer0_4value=-5., rew_a_x0_4value=-0.05,
                         punish_veh2veh4training=1., punish_veh2road4training=1., punish_veh2bike4training=1.,
                         punish_veh2person4training=1., punish_veh2speed4training=1.,
                         punish_veh2veh4real=1., punish_veh2road4real=1., punish_veh2bike4real=1.,
                         punish_veh2person4real=1., punish_veh2speed4real=1.)

            rewards = scale['rew_devi_v'] * rew_devi_v + scale['rew_devi_lateral'] * rew_devi_lateral + \
                      scale['rew_devi_phi'] * rew_devi_phi + \
                      scale['rew_yaw_rate'] * rew_yaw_rate + scale['rew_vy'] * rew_vy +\
                      scale['rew_steer0'] * rew_steer0 + scale['rew_a_x0'] * rew_a_x0 + \
                      scale['rew_steer1'] * rew_steer1 + scale['rew_a_x1'] * rew_a_x1

            rewards4value = scale['rew_devi_v_4value'] * rew_devi_v + scale['rew_devi_lateral_4value'] * rew_devi_lateral + \
                            scale['rew_devi_phi_4value'] * rew_devi_phi + \
                            scale['rew_yaw_rate_4value'] * rew_yaw_rate + scale['rew_vy_4value'] * rew_vy + \
                            scale['rew_steer0_4value'] * rew_steer0 + scale['rew_a_x0_4value'] * rew_a_x0  # TODO(guanyang): consistent with c++ code

            punish = scale['punish_veh2veh4training'] * punish_veh2veh4training + \
                     scale['punish_veh2road4training'] * punish_veh2road4training + \
                     scale['punish_veh2bike4training'] * punish_veh2bike4training + \
                     scale['punish_veh2person4training'] * punish_veh2person4training + \
                     scale['punish_veh2speed4training'] * punish_veh2speed4training
            real_punish_term = scale['punish_veh2veh4real'] * punish_veh2veh4real + \
                               scale['punish_veh2road4real'] * punish_veh2road4real + \
                               scale['punish_veh2bike4real'] * punish_veh2bike4real + \
                               scale['punish_veh2person4real'] * punish_veh2person4real + \
                               scale['punish_veh2speed4real'] * punish_veh2speed4real

            reward_dict = dict(rewards=rewards,
                               rewards4value=rewards4value,
                               punish=punish,
                               real_punish_term=real_punish_term,
                               rew_devi_v=rew_devi_v,
                               rew_devi_lateral=rew_devi_lateral,
                               rew_devi_phi=rew_devi_phi,
                               rew_yaw_rate=rew_yaw_rate,
                               rew_steer0=rew_steer0,
                               rew_a_x0=rew_a_x0,
                               rew_steer1=rew_steer1,
                               rew_a_x1=rew_a_x1,
                               punish_veh2veh4training=punish_veh2veh4training,
                               punish_veh2road4training=punish_veh2road4training,
                               punish_veh2bike4training=punish_veh2bike4training,
                               punish_veh2person4training=punish_veh2person4training,
                               punish_veh2speed4training=punish_veh2speed4training,
                               punish_veh2veh4real=punish_veh2veh4real,
                               punish_veh2road4real=punish_veh2road4real,
                               punish_veh2bike4real=punish_veh2bike4real,
                               punish_veh2person4real=punish_veh2person4real,
                               punish_veh2speed4real=punish_veh2speed4real,
                               scaled_rew_devi_v=scale['rew_devi_v'] * rew_devi_v,
                               scaled_rew_devi_lateral=scale['rew_devi_lateral'] * rew_devi_lateral,
                               scaled_rew_devi_phi=scale['rew_devi_phi'] * rew_devi_phi,
                               scaled_rew_yaw_rate=scale['rew_yaw_rate'] * rew_yaw_rate,
                               scaled_rew_vy=scale['rew_vy'] * rew_vy,
                               scaled_rew_steer0=scale['rew_steer0'] * rew_steer0,
                               scaled_rew_a_x0=scale['rew_a_x0'] * rew_a_x0,
                               scaled_rew_steer1=scale['rew_steer1'] * rew_steer1,
                               scaled_rew_a_x1=scale['rew_a_x1'] * rew_a_x1,
                               scaled_punish_veh2veh4training=scale['punish_veh2veh4training'] * punish_veh2veh4training,
                               scaled_punish_veh2road4training=scale['punish_veh2road4training'] * punish_veh2road4training,
                               scaled_punish_veh2bike4training=scale['punish_veh2bike4training'] * punish_veh2bike4training,
                               scaled_punish_veh2person4training=scale['punish_veh2person4training'] * punish_veh2person4training,
                               scaled_punish_veh2speed4training=scale['punish_veh2speed4training'] * punish_veh2speed4training,
                               )
            return reward_dict

    def compute_next_obses(self, obses, actions, untransformed_actions):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)
        obses_other = tf.stop_gradient(obses_other)
        next_obses_ego = self._ego_predict(obses_ego, actions)
        next_obses_track = self._compute_next_track_info_vectorized(next_obses_ego)
        next_obses_his_ac = tf.concat([obses_his_ac[:, -2:], untransformed_actions], axis=-1)
        next_obses_other = self._other_predict(obses_other)
        next_obses = tf.concat([next_obses_ego, next_obses_track, obses_light, obses_task, obses_ref, next_obses_his_ac,
                                next_obses_other],
                               axis=-1)
        next_obses = self._convert_to_rela(next_obses)
        return next_obses

    def _find_closest_point_batch(self, xs, ys, paths):
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), [1, self.path_len])
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), [1, self.path_len])
        pathx_tile = paths[:, 0, :]
        pathy_tile = paths[:, 1, :]
        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)
        indexs = tf.argmin(dist_array, 1)
        ref_points = tf.gather(paths, indices=indexs, axis=-1, batch_dims=1)
        return indexs, ref_points

    def _compute_next_track_info_vectorized(self, next_ego_infos):
        ego_vxs, ego_vys, ego_rs, ego_xs, ego_ys, ego_phis = [next_ego_infos[:, i] for i in range(self.ego_info_dim)]

        # find close point
        indexes, ref_points = self._find_closest_point_batch(ego_xs, ego_ys, self.future_n_points)

        ref_xs, ref_ys, ref_phis, ref_vs = [ref_points[:, i] for i in range(4)]
        ref_phis_rad = ref_phis * np.pi / 180

        vector_ref_phi = tf.stack([tf.cos(ref_phis_rad), tf.sin(ref_phis_rad)], axis=-1)
        vector_ref_phi_ccw_90 = tf.stack([-tf.sin(ref_phis_rad), tf.cos(ref_phis_rad)],
                                         axis=-1)  # ccw for counterclockwise
        vector_ego2ref = tf.stack([ref_xs - ego_xs, ref_ys - ego_ys], axis=-1)

        signed_dist_longi = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi, axis=-1))
        signed_dist_lateral = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi_ccw_90, axis=-1))

        delta_phi = deal_with_phi_diff(ego_phis - ref_phis)
        delta_vs = ego_vxs - ref_vs
        return tf.stack([signed_dist_lateral, delta_phi, delta_vs], axis=-1)

    def _convert_to_rela(self, obses):
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat(
            [tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_other_info_dim - 2))],
            axis=-1)
        ego = tf.expand_dims(ego, 1)
        rela = obses_other_reshape - ego
        rela_obses_other = self._reshape_other(rela, reverse=True)
        return tf.concat([obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, rela_obses_other],
                         axis=-1)

    def _convert_to_abso(self, rela_obses):
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other =\
            self._split_all(rela_obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat(
            [tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_other_info_dim - 2))],
            axis=-1)
        ego = tf.expand_dims(ego, 1)
        abso = obses_other_reshape + ego
        abso_obses_other = self._reshape_other(abso, reverse=True)

        return tf.concat([obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, abso_obses_other],
                         axis=-1)

    def _ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis = ego_next_infos[:, 0], ego_next_infos[:, 1], ego_next_infos[:, 2], \
                                       ego_next_infos[:, 3], ego_next_infos[:, 4], ego_next_infos[:, 5]
        v_xs = tf.clip_by_value(v_xs, 0., 35.)
        ego_next_infos = tf.stack([v_xs, v_ys, rs, xs, ys, phis], axis=-1)
        return ego_next_infos

    def _other_predict(self, obses_other):
        obses_other_reshape = self._reshape_other(obses_other)

        xs, ys, vs, phis, turn_rad = obses_other_reshape[:, :, 0], obses_other_reshape[:, :, 1], \
                                     obses_other_reshape[:, :, 2], obses_other_reshape[:, :, 3], \
                                     obses_other_reshape[:, :, -1]
        phis_rad = phis * np.pi / 180.

        xs_delta = vs / self.base_frequency * tf.cos(phis_rad)
        ys_delta = vs / self.base_frequency * tf.sin(phis_rad)
        phis_rad_delta = vs / self.base_frequency * turn_rad

        next_xs, next_ys, next_vs, next_phis_rad = xs + xs_delta, ys + ys_delta, vs, phis_rad + phis_rad_delta
        next_phis_rad = tf.where(next_phis_rad > np.pi, next_phis_rad - 2 * np.pi, next_phis_rad)
        next_phis_rad = tf.where(next_phis_rad <= -np.pi, next_phis_rad + 2 * np.pi, next_phis_rad)
        next_phis = next_phis_rad * 180 / np.pi
        next_info = tf.concat([tf.stack([next_xs, next_ys, next_vs, next_phis], -1), obses_other_reshape[:, :, 4:]],
                              axis=-1)
        next_obses_other = self._reshape_other(next_info, reverse=True)
        return next_obses_other

    def _split_all(self, obses):
        obses_ego = obses[:, :self.ego_info_dim]
        obses_track = obses[:, self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        obses_light = obses[:, self.ego_info_dim + self.track_info_dim:
                               self.ego_info_dim + self.track_info_dim + self.light_info_dim]
        obses_task = obses[:, self.ego_info_dim + self.track_info_dim + self.light_info_dim:
                              self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim]
        obses_ref = obses[:, self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim:
                             self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim + self.ref_info_dim]
        obses_his_ac = obses[:,
                       self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim + self.ref_info_dim:
                       self.other_start_dim]
        obses_other = obses[:, self.other_start_dim:]

        return obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other

    def _split_other(self, obses_other):
        obses_bike = obses_other[:, :self.bike_num * self.per_other_info_dim]
        obses_person = obses_other[:, self.bike_num * self.per_other_info_dim:
                                      (self.bike_num + self.person_num) * self.per_other_info_dim]
        obses_veh = obses_other[:, (self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obses_bike, obses_person, obses_veh

    def _reshape_other(self, obses_other, reverse=False):
        if reverse:
            return tf.reshape(obses_other, (-1, self.other_number * self.per_other_info_dim))
        else:
            return tf.reshape(obses_other, (-1, self.other_number, self.per_other_info_dim))

    def _action_transformation_for_end2end(self, actions):  # [-1, 1]
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = Para.STEER_SCALE * steer_norm, Para.ACC_SCALE * a_xs_norm + Para.ACC_SHIFT
        return tf.stack([steer_scale, a_xs_scale], 1)

    def render(self):
        abs_obses = self._convert_to_abso(self.obses).numpy()
        obses_ego, obses_track, obses_light, obses_task, obses_ref, \
        obses_his_ac, obses_other = self._split_all(abs_obses)
        all_other = []
        for index in range(Para.MAX_OTHER_NUM):
            item = obses_other[0, Para.PER_OTHER_INFO_DIM * index:Para.PER_OTHER_INFO_DIM * (index + 1)]
            other_x, other_y, other_v, other_phi, other_l, other_w = item[0], item[1], item[2], item[3], item[4], item[5]
            if index < Para.MAX_BIKE_NUM:
                other_type = 'bicycle_1'
            elif Para.MAX_BIKE_NUM <= index < Para.MAX_PERSON_NUM + Para.MAX_BIKE_NUM:
                other_type = 'DEFAULT_PEDTYPE'
            else:
                other_type = 'veh'
            all_other.append({'x': other_x, 'y': other_y, 'v': other_v, 'phi': other_phi,
                              'l': other_l, 'w': other_w, 'type':other_type})
        action = self.actions[0].numpy()
        reward_dict = {}
        for k, v in self.reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        render(light_phase=None, all_other=all_other, interested_other=None, attn_weights=None,
               obs=self.obses.numpy()[0], ref_path=None, future_n_point=None, action=action,
               done_type=None, reward_info=reward_dict, hist_posi=None, path_values=None)
        plt.show()
        plt.pause(0.001)


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class ReferencePath(object):
    def __init__(self, task, green_or_red='green', path_index=None):
        self.task = task
        self.path_list = {}
        self.path_len_list = []
        self.control_points = []
        self._construct_ref_path(self.task)
        self.path = None
        self.path_index = None
        self.ref_encoding = None
        self.set_path(green_or_red, path_index)

    def set_path(self, green_or_red='green', path_index=None):
        if path_index is None:
            path_index = np.random.choice(len(self.path_list[green_or_red]))
        self.path_index = path_index
        self.ref_encoding = REF_ENCODING[path_index]
        self.path = self.path_list[green_or_red][path_index]

    def get_future_n_point(self, ego_x, ego_y, n, dt=0.1):  # not include the current closest point
        idx, _ = self._find_closest_point(ego_x, ego_y)
        future_n_x, future_n_y, future_n_phi, future_n_v = [], [], [], []
        for _ in range(n):
            x, y, phi, v = self.idx2point(idx)
            ds = v * dt
            s = 0
            while s < ds:
                if idx + 1 >= len(self.path[0]):
                    break
                next_x, next_y, _, _ = self.idx2point(idx + 1)
                s += np.sqrt(np.square(next_x - x) + np.square(next_y - y))
                x, y = next_x, next_y
                idx += 1
            x, y, phi, v = self.idx2point(idx)
            future_n_x.append(x)
            future_n_y.append(y)
            future_n_phi.append(phi)
            future_n_v.append(v)
        future_n_point = np.stack([np.array(future_n_x, dtype=np.float32), np.array(future_n_y, dtype=np.float32),
                                   np.array(future_n_phi, dtype=np.float32), np.array(future_n_v, dtype=np.float32)],
                                  axis=0)
        return future_n_point

    def tracking_error_vector_vectorized(self, ego_x, ego_y, ego_phi, ego_v):
        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        phi0_rad = phi0 * np.pi / 180
        vector_ref_phi = np.array([np.cos(phi0_rad), np.sin(phi0_rad)])
        vector_ref_phi_ccw_90 = np.array([-np.sin(phi0_rad), np.cos(phi0_rad)])  # ccw for counterclockwise
        vector_ego2ref = np.array([x0 - ego_x, y0 - ego_y])

        signed_dist_longi = np.negative(np.dot(vector_ego2ref, vector_ref_phi))
        signed_dist_lateral = np.negative(np.dot(vector_ego2ref, vector_ref_phi_ccw_90))

        return np.array([signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0])

    def idx2point(self, idx):
        return self.path[0][idx], self.path[1][idx], self.path[2][idx], self.path[3][idx]

    def _construct_ref_path(self, task):
        sl = 40  # straight length
        dece_dist = 20
        meter_pointnum_ratio = 30
        planed_trj_g = []
        planed_trj_r = []
        if task == 'left':
            end_offsets = [Para.LANE_WIDTH * (i + 0.5) for i in range(Para.LANE_NUMBER)]
            start_offsets = [Para.LANE_WIDTH * 0.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE / 2
                    control_point4 = -Para.CROSSROAD_SIZE / 2, end_offset
                    control_point2, control_point3 = \
                        get_bezier_control_points(control_point1[0], control_point1[1], 0.5 * pi,
                                                  control_point4[0], control_point4[1], pi)
                    self.control_points.append([control_point1, control_point2, control_point3, control_point4])

                    node = np.asfortranarray(
                        [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                         [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                        dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = Para.LANE_WIDTH / 2 * np.ones(shape=(sl * meter_pointnum_ratio,),
                                                                          dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE / 2 - sl, -Para.CROSSROAD_SIZE / 2,
                                                        sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(-Para.CROSSROAD_SIZE / 2, -Para.CROSSROAD_SIZE / 2 - sl,
                                                      sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
                    vs_green = np.array([6.66] * len(start_straight_line_x) + [4.16] * (len(trj_data[0]) - 1) + [6.66] *
                                        len(end_straight_line_x), dtype=np.float32)
                    vs_red_0 = np.array([6.66] * (
                            len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist)), dtype=np.float32)
                    vs_red_1 = np.linspace(6.66, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                    vs_red_2 = np.array(
                        [0.0] * (len(xs_1) - len(vs_red_0) - len(vs_red_1)), dtype=np.float32)
                    vs_red = np.append(np.append(vs_red_0, vs_red_1), vs_red_2)
                    planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    planed_trj_red = xs_1, ys_1, phis_1, vs_red
                    planed_trj_g.append(planed_trj_green)
                    planed_trj_r.append(planed_trj_red)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}
        elif task == 'straight':
            end_offsets = [Para.LANE_WIDTH * (i + 0.5) for i in range(Para.LANE_NUMBER)]
            start_offsets = [Para.LANE_WIDTH * 1.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE / 2
                    control_point4 = end_offset, Para.CROSSROAD_SIZE / 2
                    control_point2, control_point3 = \
                        get_bezier_control_points(control_point1[0], control_point1[1], 0.5 * pi,
                                                  control_point4[0], control_point4[1], 0.5 * pi)
                    self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                    node = np.asfortranarray(
                        [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                         [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                        dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,),
                                                                   dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE / 2 - sl, -Para.CROSSROAD_SIZE / 2,
                                                        sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    end_straight_line_y = np.linspace(Para.CROSSROAD_SIZE / 2, Para.CROSSROAD_SIZE / 2 + sl,
                                                      sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
                    vs_green = np.array([6.66] * len(start_straight_line_x) + [4.16] * (len(trj_data[0]) - 1) + [6.66] *
                                        len(end_straight_line_x), dtype=np.float32)
                    vs_red_0 = np.array([6.66] * (
                            len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist)), dtype=np.float32)
                    vs_red_1 = np.linspace(6.66, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                    vs_red_2 = np.array(
                        [0.0] * (len(xs_1) - len(vs_red_0) - len(vs_red_1)), dtype=np.float32)
                    vs_red = np.append(np.append(vs_red_0, vs_red_1), vs_red_2)
                    planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    planed_trj_red = xs_1, ys_1, phis_1, vs_red
                    planed_trj_g.append(planed_trj_green)
                    planed_trj_r.append(planed_trj_red)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        else:
            assert task == 'right'
            end_offsets = [-Para.LANE_WIDTH * 2.5, -Para.LANE_WIDTH * 1.5, -Para.LANE_WIDTH * 0.5]
            start_offsets = [Para.LANE_WIDTH * 2.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE / 2
                    control_point4 = Para.CROSSROAD_SIZE / 2, end_offset
                    control_point2, control_point3 = \
                        get_bezier_control_points(control_point1[0], control_point1[1], 0.5 * pi,
                                                  control_point4[0], control_point4[1], 0.)
                    self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                    node = np.asfortranarray(
                        [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                         [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                        dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,),
                                                                   dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE / 2 - sl, -Para.CROSSROAD_SIZE / 2,
                                                        sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(Para.CROSSROAD_SIZE / 2, Para.CROSSROAD_SIZE / 2 + sl,
                                                      sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
                    vs_green = np.array([6.66] * len(start_straight_line_x) + [4.16] * (len(trj_data[0]) - 1) + [6.66] *
                                        len(end_straight_line_x), dtype=np.float32)  # TODO(guanyang): check its length
                    planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    planed_trj_red = xs_1, ys_1, phis_1, vs_green  # the same velocity design for turning right
                    planed_trj_g.append(planed_trj_green)
                    planed_trj_r.append(planed_trj_red)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

    def _find_closest_point(self, x, y, ratio=10):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]
        dists = np.square(x - reduced_path_x) + np.square(y - reduced_path_y)
        idx = np.argmin(dists) * ratio
        return idx, self.idx2point(idx)

    def plot_path(self, x, y):
        plt.axis('equal')
        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, path in enumerate(self.path_list['green']):
            plt.plot(path[0], path[1], color=color[i], alpha=1.0)

        for _, point in enumerate(self.control_points):
            for item in point:
                plt.scatter(item[0], item[1], color='red')
        print(self.path_len_list)

        index, closest_point = self._find_closest_point(np.array([x], np.float32),
                                                        np.array([y], np.float32))
        plt.plot(x, y, 'b*')
        plt.plot(closest_point[0], closest_point[1], 'ro')
        plt.show()


def test_ref_path():
    path = ReferencePath('right')
    path.plot_path(1.875, 0)


def test_future_n_data():
    path = ReferencePath('straight')
    plt.axis('equal')
    current_i = 600
    plt.plot(path.path[0], path.path[1])
    future_data_list = path.future_n_data(current_i, 5)
    plt.plot(path.indexs2points(current_i)[0], path.indexs2points(current_i)[1], 'go')
    for point in future_data_list:
        plt.plot(point[0], point[1], 'r*')
    plt.show()


def test_tracking_error_vector():
    path = ReferencePath('straight')
    xs = np.array([1.875, 1.875, -10, -20], np.float32)
    ys = np.array([-20, 0, -10, -1], np.float32)
    phis = np.array([90, 135, 135, 180], np.float32)
    vs = np.array([10, 12, 10, 10], np.float32)

    tracking_error_vector = path.tracking_error_vector(xs, ys, phis, vs, 10)
    print(tracking_error_vector)


def test_model():
    from endtoend import CrossroadEnd2end
    env = CrossroadEnd2end('left', 0)
    model = EnvironmentModel('left', 0)
    obs_list = []
    obs = env.reset()
    done = 0
    # while not done:
    for i in range(10):
        obs_list.append(obs)
        action = np.array([0, -1], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        env.render()
    obses = np.stack(obs_list, 0)
    model.reset(obses, 'left')
    print(obses.shape)
    for rollout_step in range(100):
        actions = tf.tile(tf.constant([[0.5, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
        obses, rewards, punish1, punish2, _, _ = model.rollout(actions)
        print(rewards.numpy()[0], punish1.numpy()[0])
        model.render()


def test_tf_function():
    class Test2():
        def __init__(self):
            self.c = 2

        def step1(self, a):
            print('trace')
            self.c = a

        def step2(self):
            return self.c

    test2 = Test2()

    @tf.function  # (input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def f(a):
        test2.step1(a)
        return test2.step2()

    print(f(2), type(test2.c))
    print(f(2), test2.c)

    print(f(tf.constant(2)), type(test2.c))
    print(f(tf.constant(3)), test2.c)

    # print(f(2), test2.c)
    # print(f(3), test2.c)
    # print(f(2), test2.c)
    # print(f())
    # print(f())
    #
    # test2.c.assign_add(12)
    # print(test2.c)
    # print(f())

    # b= test2.create_test1(1)
    # print(test2.b,b, test2.b.a)
    # b=test2.create_test1(2)
    # print(test2.b,b,test2.b.a)
    # b=test2.create_test1(1)
    # print(test2.b,b,test2.b.a)
    # test2.create_test1(1)
    # test2.pc()
    # test2.create_test1(1)
    # test2.pc()


@tf.function
def test_tffunc(inttt):
    print(22)
    if inttt == '1':
        a = 2
    elif inttt == '2':
        a = 233
    else:
        a = 22
    return a


def test_ref():
    import numpy as np
    import matplotlib.pyplot as plt
    ref = ReferencePath('left', 'green')
    print('left path:', len(ref.path_list['green'][0][0]), len(ref.path_list['green'][1][0]),
          len(ref.path_list['green'][2][0]))

    ref = ReferencePath('straight', 'green')
    print('straight path:', len(ref.path_list['green'][0][0]), len(ref.path_list['green'][1][0]),
          len(ref.path_list['green'][2][0]))

    ref = ReferencePath('right', 'green')
    print('right path:', len(ref.path_list['green'][0][0]), len(ref.path_list['green'][1][0]),
          len(ref.path_list['green'][2][0]))


if __name__ == '__main__':
    test_ref()
