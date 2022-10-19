import numpy as np
import tensorflow as tf
from env_and_model.multi_lane.car_tracking.utils import *
import math
from math import tan
from env_and_model.multi_lane.car_tracking.ref_path import ReferencePath
# from env_and_model.multi_lane.multilane import MultiLane
# tf.enable_eager_execution()


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

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
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.cast(tf.zeros_like(a_x), dtype=tf.float32))
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


class MultiLaneModel(object):
    def __init__(self):
        self.ego_dim = Para.EGO_DIM
        self.track_dim = 3
        self.closest_point_dim = 3
        self.ref_dim = Para.N * 3
        self.per_ref_dim = 3
        self.veh_dim = 6 * 4
        self.path_len = Para.N
        self.veh_num = 6
        self.per_veh_info_dim = 4
        self.base_frequency = 10

        self.ref_v = 5
        self.vehicle_dynamics = VehicleDynamics()
        self.actions = None
        self.reward_dict = None
        self.obses = None

    def rollout(self, actions):
        self.actions = self._action_transformation_for_end2end(actions)
        rewards, self.reward_dict = self.compute_rewards(self.obses, self.actions)
        self.obses = self.compute_next_obses(self.obses, self.actions)
        # self.render()
        return self.obses, self.reward_dict

    def reset(self, obses):
        self.obses = obses
        self.actions = None
        self.reward_dict = None

    def compute_rewards(self, obses, actions):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_closest_point, obses_veh, obses_ref, \
        obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v = \
            self.split_all(obses)

        with tf.name_scope('compute_reward') as scope:
            veh_info = tf.stop_gradient(obses_veh)
            # ref_info = tf.stop_gradient(obses_ref)
            # left_lane_info = tf.stop_gradient(obses_left_lane)
            # right_lane_info = tf.stop_gradient(obses_right_lane)
            # lane_width_info = tf.stop_gradient(obses_lane_width)

            steers, a_xs = actions[:, 0], actions[:, 1]

            # rewards related to tracking error
            rew_devi_lateral = -tf.cast(tf.square(obses_track[:, 0]), dtype=tf.float32)
            rew_devi_phi = -tf.cast(tf.square(obses_track[:, 1] * np.pi / 180.), dtype=tf.float32)
            rew_devi_v = -tf.cast(tf.square(obses_track[:, 2]), dtype=tf.float32)

            # rewards related to ego stability
            punish_yaw_rate = tf.square(obses_ego[:, 2])
            punish_yaw_rate = tf.cast(punish_yaw_rate, tf.float32)

            # rewards related to action
            punish_steer = tf.square(steers)
            punish_a_x = tf.square(a_xs)
            # punish_steer = tf.cast(punish_steer, tf.float64)
            # punish_a_x = tf.cast(punish_a_x, tf.float64)

            # veh2veh punishment
            ego_lws = (Para.L - Para.W) / 2.
            ego_front_points = tf.cast(obses_ego[:, 3] + ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32), \
                               tf.cast(obses_ego[:, 4] + ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32)
            ego_rear_points = tf.cast(obses_ego[:, 3] - ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32), \
                              tf.cast(obses_ego[:, 4] - ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32)

            veh2veh4real = tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32)
            veh2veh4training = tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32)
            # veh2road4real = tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32)
            veh2road4training = tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32)

            for veh_index in range(6):
                vehs = veh_info[:, (veh_index * 4): (veh_index + 1) * 4]
                veh_lw = (Para.L - Para.W) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lw * tf.cos(vehs[:, 2] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lw * tf.sin(vehs[:, 2] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lw * tf.cos(vehs[:, 2] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lw * tf.sin(vehs[:, 2] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(
                            tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
                        veh2veh4training += tf.where(veh2veh_dist - 3.5 < 0, tf.square(veh2veh_dist - 3.5),
                                                     tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32))
                        veh2veh4real += tf.where(veh2veh_dist - 2.5 < 0, tf.square(veh2veh_dist - 2.5),
                                                 tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32))

            # veh2road punishment
            for ego_points in [ego_front_points, ego_rear_points]:
                k = tf.cast((tf.tan(obses_closest_point[:, 2] * np.pi / 180)), dtype=tf.float32)
                # straigth line
                veh2line_1 = tf.sqrt(tf.square(ego_points[0] - obses_closest_point[:, 0]))
                is_left_1 = (ego_points[0] < obses_closest_point[:, 0])
                # true v2l
                b = tf.cast((obses_closest_point[:, 1] - k * obses_closest_point[:, 0]), dtype=tf.float32)
                x = (k * ego_points[1] + ego_points[0] - k * b) / (tf.square(k) + 1)
                y = (k ** 2 * ego_points[1] + k * ego_points[0] + b) / (k ** 2 + 1)
                veh2line_2 = tf.sqrt(tf.square(ego_points[0] - x) + tf.square(ego_points[1] - y))
                is_left_2 = judge_point_line_pos(ego_points, k, obses_closest_point[:, 0], obses_closest_point[:, 1])

                veh2line = tf.where(tf.abs(k) - 1000 > 0, veh2line_1, veh2line_2)
                is_left = tf.where(tf.abs(k) - 1000 > 0, is_left_1, is_left_2)
                # veh2line = veh2line_1
                # is_left = is_left_1


                left_dist = tf.where((0.5 + obses_left_lane[:, 0]) * obses_lane_width[:, 0] - veh2line - 1.25 < 0,
                                     tf.square((0.5 + obses_left_lane[:, 0]) * obses_lane_width[:, 0] - veh2line - 1.25),
                                     tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32))
                right_dist = tf.where((0.5 + obses_right_lane[:, 0]) * obses_lane_width[:, 0] - veh2line - 1.25 < 0,
                                      tf.square((0.5 + obses_right_lane[:, 0]) * obses_lane_width[:, 0] - veh2line - 1.25),
                                      tf.cast(tf.zeros_like(obses[:, 0]), dtype=tf.float32))
                veh2road4training += tf.where(is_left, left_dist, right_dist)

            rewards = Para.scale_devi_p * rew_devi_lateral + \
                      Para.scale_devi_v * rew_devi_v + \
                      Para.scale_devi_phi * rew_devi_phi + \
                      Para.scale_punish_yaw_rate * punish_yaw_rate + \
                      Para.scale_punish_steer * punish_steer + \
                      Para.scale_punish_a_x * punish_a_x + \
                      Para.scale_devi_p * 0.25 + \
                      Para.scale_devi_v * 25 + \
                      Para.scale_devi_phi * np.square(30 / 180 * np.pi)

            punish_term_for_training = veh2veh4training + veh2road4training

            rewards -= Para.punish_factor * punish_term_for_training

            reward_dict = dict(rewards=rewards,
                               rewards4value=rewards,
                               punish=punish_term_for_training,
                               rew_devi_v=rew_devi_v,
                               rew_devi_lateral=rew_devi_lateral,
                               rew_devi_phi=rew_devi_phi,
                               # rew_yaw_rate=rew_yaw_rate,
                               # rew_steer0=rew_steer0,
                               # rew_a_x0=rew_a_x0,
                               # rew_steer1=rew_steer1,
                               # rew_a_x1=rew_a_x1,
                               punish_steer=punish_steer,
                               punish_a_x=punish_a_x,
                               punish_yaw_rate=punish_yaw_rate,
                               scaled_devi_p=Para.scale_devi_p * rew_devi_lateral,
                               scaled_devi_v=Para.scale_devi_v * rew_devi_v,
                               scaled_devi_phi=Para.scale_devi_phi * rew_devi_phi,
                               scaled_punish_steer=Para.scale_punish_steer * punish_steer,
                               scaled_punish_a_x=Para.scale_punish_a_x * punish_a_x,
                               scaled_punish_yaw_rate=Para.scale_punish_yaw_rate * punish_yaw_rate,
                               punish_term_for_training=punish_term_for_training,
                               veh2veh4training=veh2veh4training,
                               veh2road4training=veh2road4training
                               )

            return rewards, reward_dict

    def compute_next_obses(self, obses, actions):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_closest_point, obses_veh, obses_ref,\
        obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v = \
            self.split_all(obses)
        obses_veh = tf.stop_gradient(obses_veh)
        next_obses_ego = self._ego_predict(obses_ego, actions)
        next_obses_veh = self._veh_predict(obses_veh)
        ref_n_point_batch = tf.transpose(tf.reshape(obses_ref, shape=(-1, self.path_len, self.per_ref_dim)), perm=[0, 2, 1])
        next_obses_closest_point = self._find_closest_point_batch(next_obses_ego[:, 3], next_obses_ego[:, 4], ref_n_point_batch)
        next_obses_track = self._compute_next_track_vector(next_obses_ego, next_obses_closest_point, obses_ref_v)
        next_obses = tf.concat([next_obses_ego, next_obses_track, next_obses_closest_point,
                                next_obses_veh, obses_ref, obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v],
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
        k = tf.tan(ref_points[:, 2] * np.pi / 180)
        b = ref_points[:, 1] - k * ref_points[:, 0]
        x = (k * ys + xs - k * b) / (k ** 2 + 1)
        y = (k ** 2 * ys + k * xs + b) / (k ** 2 + 1)
        cl_x = tf.where(abs(k) > 1000, ref_points[:, 0], x)
        cl_y = tf.where(abs(k) > 1000, ys, y)
        return tf.stack([cl_x, cl_y, ref_points[:, 2]], axis=1)

    def _compute_next_track_vector(self, next_obses_ego, next_obses_closest_point, next_obses_ref_v):
        ego_vxs, ego_vys, ego_rs, ego_xs, ego_ys, ego_phis = [next_obses_ego[:, i] for i in range(self.ego_dim)]
        ref_xs, ref_ys, ref_phis = [next_obses_closest_point[:, i] for i in range(3)]
        ref_phis_rad = ref_phis * np.pi / 180
        vector_ref_phi = tf.stack([tf.cos(ref_phis_rad), tf.sin(ref_phis_rad)], axis=-1)
        vector_ref_phi_ccw_90 = tf.stack([-tf.sin(ref_phis_rad), tf.cos(ref_phis_rad)],
                                         axis=-1)  # ccw for counterclockwise
        vector_ego2ref = tf.stack([ref_xs - ego_xs, ref_ys - ego_ys], axis=-1)

        signed_dist_longi = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi, axis=-1))
        signed_dist_lateral = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi_ccw_90, axis=-1))

        delta_phi = deal_with_phi_diff(ego_phis - ref_phis)
        delta_vs = ego_vxs - next_obses_ref_v[:, 0]
        return tf.stack([signed_dist_lateral, delta_phi, delta_vs], axis=-1)

    def _convert_to_rela(self, obses):
        obses_ego, obses_track, obses_closest_point, obses_veh, obses_ref, \
        obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v = \
            self.split_all(obses)
        obses_veh_reshape = self._reshape_veh(obses_veh)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat(
            [tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_veh_info_dim - 2))],
            axis=-1)
        ego = tf.expand_dims(ego, 1)
        rela = obses_veh_reshape - ego
        rela_obses_veh = self._reshape_veh(rela, reverse=True)
        return tf.concat([obses_ego, obses_track, obses_closest_point, rela_obses_veh, obses_ref,
                          obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v],
                         axis=-1)

    def _convert_to_abso(self, rela_obses):
        obses_ego, obses_track, obses_closest_point, obses_veh, obses_ref, obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v = \
            self.split_all(tf.cast(rela_obses, dtype=tf.float32))
        obses_veh_reshape = self._reshape_veh(obses_veh)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat(
            [tf.cast(tf.stack([ego_x, ego_y], axis=-1), dtype=tf.float32), tf.zeros(shape=(ego_x.shape[0], self.per_veh_info_dim - 2))],
            axis=-1)
        ego = tf.expand_dims(ego, 1)
        abso = tf.cast(obses_veh_reshape, dtype=tf.float32) + ego
        abso_obses_veh = self._reshape_veh(abso, reverse=True)
        return tf.concat([obses_ego, obses_track, obses_closest_point, abso_obses_veh, obses_ref,
                          obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v],
                         axis=-1)

    def _reshape_veh(self, obses_veh, reverse=False):
        if reverse:
            return tf.reshape(obses_veh, (-1, self.veh_num * self.per_veh_info_dim))
        else:
            return tf.reshape(obses_veh, (-1, self.veh_num, self.per_veh_info_dim))

    def _ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos, actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis = ego_next_infos[:, 0], ego_next_infos[:, 1], ego_next_infos[:, 2], \
                                       ego_next_infos[:, 3], ego_next_infos[:, 4], ego_next_infos[:, 5]
        v_xs = tf.clip_by_value(v_xs, 0., 35.)
        ego_next_infos = tf.stack([v_xs, v_ys, rs, xs, ys, phis], axis=-1)
        return ego_next_infos

    def _veh_predict(self, obses_veh):
        obses_veh_reshape = self._reshape_veh(obses_veh)

        xs, ys, phis, vs = obses_veh_reshape[:, :, 0], obses_veh_reshape[:, :, 1], \
                           obses_veh_reshape[:, :, 2], obses_veh_reshape[:, :, 3]

        phis_rad = phis * np.pi / 180.

        xs_delta = vs / self.base_frequency * tf.cos(phis_rad)
        ys_delta = vs / self.base_frequency * tf.sin(phis_rad)

        next_xs, next_ys, next_vs, next_phis_rad = xs + xs_delta, ys + ys_delta, vs, phis_rad

        # next_phis_rad = tf.where(next_phis_rad > np.pi, next_phis_rad - 2 * np.pi, next_phis_rad)
        # next_phis_rad = tf.where(next_phis_rad <= -np.pi, next_phis_rad + 2 * np.pi, next_phis_rad)

        next_phis = next_phis_rad * 180 / np.pi
        next_info = tf.stack([next_xs, next_ys, next_phis, next_vs], -1)
        next_obses_veh = self._reshape_veh(next_info, reverse=True)
        return next_obses_veh

    def split_all(self, obses):
        obses_ego = obses[:, :self.ego_dim]
        obses_track = obses[:, self.ego_dim: self.ego_dim + self.track_dim]
        obses_closest_point = obses[:, (self.ego_dim + self.track_dim):
                                       (self.ego_dim + self.track_dim + self.closest_point_dim)]
        obses_veh = obses[:, (self.ego_dim + self.track_dim + self.closest_point_dim):
                             (self.ego_dim + self.track_dim + self.closest_point_dim + self.veh_dim)]
        obses_ref = obses[:, (self.ego_dim + self.track_dim + self.closest_point_dim + self.veh_dim):
                             (self.ego_dim + self.track_dim + self.closest_point_dim + self.veh_dim + self.ref_dim)]
        obses_left_lane = obses[:, (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim):
                                   (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim + 1)]
        obses_right_lane = obses[:, (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim + 1):
                                    (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim + 2)]
        obses_lane_width = obses[:, (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim + 2):
                                    (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim + 3)]
        obses_ref_v = obses[:, (self.ego_dim + self.track_dim + self.closest_point_dim + self.ref_dim + self.veh_dim + 3):]

        return obses_ego, obses_track, obses_closest_point, obses_veh, obses_ref,\
               obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v

    def _action_transformation_for_end2end(self, actions):
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = Para.STEER_SCALE * steer_norm, Para.ACC_SCALE * a_xs_norm + Para.ACC_SHIFT
        return tf.stack([steer_scale, a_xs_scale], 1)

    def render(self):
        abs_obses = self._convert_to_abso(self.obses).numpy()
        obses_ego, obses_track, obses_closest_point, obses_veh, obses_ref, \
        obses_left_lane, obses_right_lane, obses_lane_width, obses_ref_v = \
            self.split_all(abs_obses)
        veh_xs = []
        veh_ys = []
        veh_phis = []
        for index in range(6):
            item = obses_veh[0, 4 * index: 4 * (index + 1)]
            other_l, other_w = 4, 2.4
            other_x, other_y, other_phi, other_v = item[0], item[1], item[2], item[3]
            veh_xs.append(other_x)
            veh_ys.append(other_y)
            veh_phis.append(other_phi)
        patches = []
        plt.clf()
        ax = plt.axes([0.05, 0.05, 0.9, 0.9])
        ax.axis('equal')

        # plot road typology
        ref_n_point = tf.transpose(tf.reshape(obses_ref, shape=(-1, self.path_len, self.per_ref_dim)),
                                         perm=[0, 2, 1])[0]
        ax.plot(obses_closest_point[0, 0], obses_closest_point[0, 1])
        plot_multi_lane(ax, ref_n_point[0], ref_n_point[1], ref_n_point[2],
                        obses_left_lane[0], obses_right_lane[0], obses_lane_width[0])

        # plot ego vehicle
        patches.append(
            draw_rotate_rec(obses_ego[0, -3], obses_ego[0, -2], obses_ego[0, -1], Para.L, Para.W)
        )

        # plot sur vehicle
        patches.extend(draw_rotate_batch_rec(veh_xs, veh_ys, veh_phis, Para.L, Para.W))

        # plot closest point
        ax.scatter(obses_closest_point[0, 0], obses_closest_point[0, 1], color='r')

        # plot the whole fig
        ax.add_collection(PatchCollection(patches, match_original=True))
        plt.show()
        plt.pause(0.001)


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff

# def test_model():
#     env_data = MultiLane()
#     env_model = MultiLaneModel()
#     obs = env_data.reset()
#     obses = tf.expand_dims(obs, axis=0)
#     actions = tf.zeros((1, 2))
#     print(env_model.compute_next_obses(obses, actions))


# if __name__ == '__main__':
#     test_model()