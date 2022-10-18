#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/02/24
# @Author  : Yang Guan, Yangang Ren (Tsinghua Univ.)
# @FileName: mpc_ipopt.py
# @Function: compare adp and mpc
# =====================================
import argparse
import copy
import json
import random
import time
import datetime

import ray
import tensorflow as tf
from casadi import *

from env_and_model.idc_virtual_veh.dynamics_and_models import ReferencePath
from env_and_model.idc_virtual_veh.endtoend import IdcVirtualVehEnv
from env_and_model.idc_virtual_veh.endtoend_env_utils import *
from env_and_model.idc_virtual_veh.hierarchical_decision.multi_path_generator import MultiPathGenerator
from env_and_model.idc_virtual_veh.utils.load_policy import LoadPolicy, get_args
from env_and_model.idc_virtual_veh.utils.misc import TimerStat
from utils import *


def deal_with_phi_casa(phi):
    phi = if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))
    return phi


def deal_with_phi(phi):
    phi = if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))
    return phi


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, x, u, tau):
        v_x, v_y, r, x, y, phi = x[0], x[1], x[2], x[3], x[4], x[5]
        phi = phi * np.pi / 180.
        steer, a_x = u[0], u[1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return next_state


class Dynamics(object):
    def __init__(self, x_init, task, ref_index, tau):
        self.x_init = x_init
        self.task = task
        self.ref_index = ref_index
        self.tau = tau
        self.vd = VehicleDynamics()
        self.others = x_init[Para.OTHER_START_DIM:]
        path = ReferencePath(task, 'green')  # do not use mpc in red light case
        self.path = path.path_list['green'][self.ref_index]
        x, y, phi, _ = [ite[1200:-1200] for ite in self.path]
        self.start_exp_v, self.middle_exp_v, self.end_exp_v = \
            self.path[3][0], self.path[3][1300], self.path[3][-1]
        if self.task == 'left':
            self.start, self.end = x[0], y[-1]
            fit_x = np.arctan2(y - (-Para.CROSSROAD_SIZE / 2), x - (-Para.CROSSROAD_SIZE / 2))
            fit_y1 = np.sqrt(np.square(x - (-Para.CROSSROAD_SIZE / 2)) + np.square(y - (-Para.CROSSROAD_SIZE / 2)))
            fit_y2 = phi
        elif self.task == 'straight':
            self.start, self.end = x[0], x[-1]
            fit_x = y
            fit_y1 = x
            fit_y2 = phi
        else:
            self.start, self.end = x[0], y[-1]
            fit_x = np.arctan2(y - (-Para.CROSSROAD_SIZE / 2), x - (Para.CROSSROAD_SIZE / 2))
            fit_y1 = np.sqrt(np.square(x - (Para.CROSSROAD_SIZE / 2)) + np.square(y - (-Para.CROSSROAD_SIZE / 2)))
            fit_y2 = phi

        self.fit_y1_para = list(np.polyfit(fit_x, fit_y1, 3, rcond=None, full=False, w=None, cov=False))
        self.fit_y2_para = list(np.polyfit(fit_x, fit_y2, 3, rcond=None, full=False, w=None, cov=False))

    def tracking_error_pred(self, next_ego):
        v_x, v_y, r, x, y, phi = next_ego[0], next_ego[1], next_ego[2], next_ego[3], next_ego[4], next_ego[5]
        if self.task == 'left':
            out1 = [-(y - self.end), deal_with_phi_casa(phi - 180.), v_x - self.end_exp_v]
            out2 = [-(x - self.start), deal_with_phi_casa(phi - 90.), v_x - self.start_exp_v]
            fit_x = arctan2(y - (-Para.CROSSROAD_SIZE / 2), x - (-Para.CROSSROAD_SIZE / 2))
            ref_d = self.fit_y1_para[0] * power(fit_x, 3) + self.fit_y1_para[1] * power(fit_x, 2) + \
                    self.fit_y1_para[2] * fit_x + self.fit_y1_para[3]
            ref_phi = self.fit_y2_para[0] * power(fit_x, 3) + self.fit_y2_para[1] * power(fit_x, 2) + \
                      self.fit_y2_para[2] * fit_x + self.fit_y2_para[3]
            d = sqrt(power(x - (-Para.CROSSROAD_SIZE / 2), 2) + power(y - (-Para.CROSSROAD_SIZE / 2), 2))
            out3 = [-(d - ref_d), deal_with_phi_casa(phi - ref_phi), v_x - self.middle_exp_v]
            return [
                if_else(x < -Para.CROSSROAD_SIZE / 2, out1[0], if_else(y < -Para.CROSSROAD_SIZE / 2, out2[0], out3[0])),
                if_else(x < -Para.CROSSROAD_SIZE / 2, out1[1], if_else(y < -Para.CROSSROAD_SIZE / 2, out2[1], out3[1])),
                if_else(x < -Para.CROSSROAD_SIZE / 2, out1[2], if_else(y < -Para.CROSSROAD_SIZE / 2, out2[2], out3[2]))]
        elif self.task == 'straight':
            out1 = [-(x - self.start), deal_with_phi_casa(phi - 90.), v_x - self.start_exp_v]
            out2 = [-(x - self.end), deal_with_phi_casa(phi - 90.), v_x - self.end_exp_v]
            fit_x = y
            ref_d = self.fit_y1_para[0] * power(fit_x, 3) + self.fit_y1_para[1] * power(fit_x, 2) + \
                    self.fit_y1_para[2] * fit_x + self.fit_y1_para[3]
            ref_phi = self.fit_y2_para[0] * power(fit_x, 3) + self.fit_y2_para[1] * power(fit_x, 2) + \
                      self.fit_y2_para[2] * fit_x + self.fit_y2_para[3]
            d = x
            out3 = [-(d - ref_d), deal_with_phi_casa(phi - ref_phi), v_x - self.middle_exp_v]
            return [
                if_else(y < -Para.CROSSROAD_SIZE / 2, out1[0], if_else(y > Para.CROSSROAD_SIZE / 2, out2[0], out3[0])),
                if_else(y < -Para.CROSSROAD_SIZE / 2, out1[1], if_else(y > Para.CROSSROAD_SIZE / 2, out2[1], out3[1])),
                if_else(y < -Para.CROSSROAD_SIZE / 2, out1[2], if_else(y > Para.CROSSROAD_SIZE / 2, out2[2], out3[2]))]
        else:
            assert self.task == 'right'
            out1 = [-(x - self.start), deal_with_phi_casa(phi - 90.), v_x - self.start_exp_v]
            out2 = [y - self.end, deal_with_phi_casa(phi - 0.), v_x - self.end_exp_v]
            fit_x = arctan2(y - (-Para.CROSSROAD_SIZE / 2), x - (Para.CROSSROAD_SIZE / 2))
            ref_d = self.fit_y1_para[0] * power(fit_x, 3) + self.fit_y1_para[1] * power(fit_x, 2) + \
                    self.fit_y1_para[2] * fit_x + self.fit_y1_para[3]
            ref_phi = self.fit_y2_para[0] * power(fit_x, 3) + self.fit_y2_para[1] * power(fit_x, 2) + \
                      self.fit_y2_para[2] * fit_x + self.fit_y2_para[3]
            d = sqrt(power(x - (Para.CROSSROAD_SIZE / 2), 2) + power(y - (-Para.CROSSROAD_SIZE / 2), 2))
            out3 = [d - ref_d, deal_with_phi_casa(phi - ref_phi), v_x - self.middle_exp_v]
            return [
                if_else(y < -Para.CROSSROAD_SIZE / 2, out1[0], if_else(x > Para.CROSSROAD_SIZE / 2, out2[0], out3[0])),
                if_else(y < -Para.CROSSROAD_SIZE / 2, out1[1], if_else(x > Para.CROSSROAD_SIZE / 2, out2[1], out3[1])),
                if_else(y < -Para.CROSSROAD_SIZE / 2, out1[2], if_else(x > Para.CROSSROAD_SIZE / 2, out2[2], out3[2]))]

    def other_pred(self):
        predictions = []
        for other_idx in range(Para.MAX_OTHER_NUM):
            other = self.others[other_idx * Para.PER_OTHER_INFO_DIM:(other_idx + 1) * Para.PER_OTHER_INFO_DIM]
            x, y, v, phi, turn_rad = other[0], other[1], other[2], other[3], other[-1]
            phi_rad = phi * np.pi / 180.

            x_delta = v * self.tau * cos(phi_rad)
            y_delta = v * self.tau * sin(phi_rad)
            phi_rad_delta = v * self.tau * turn_rad

            next_x, next_y, next_v, next_phi_rad = x + x_delta, y + y_delta, v, phi_rad + phi_rad_delta
            next_phi = next_phi_rad * 180 / np.pi
            next_phi = deal_with_phi_casa(next_phi)
            predictions += [next_x, next_y, next_v, next_phi] + other[4:]
        self.others = predictions

    def f_xu(self, x, u):
        next_ego = self.vd.f_xu(x, u, self.tau)
        next_tracking = self.tracking_error_pred(next_ego)
        return next_ego + next_tracking

    def g_x(self, x):
        ego_v, ego_x, ego_y, ego_phi = x[0], x[3], x[4], x[5]
        g_list = []
        ego_lws = (Para.L - Para.W) / 2.
        ego_front_points = ego_x + ego_lws * cos(ego_phi * np.pi / 180.), \
                           ego_y + ego_lws * sin(ego_phi * np.pi / 180.)
        ego_rear_points = ego_x - ego_lws * cos(ego_phi * np.pi / 180.), \
                          ego_y - ego_lws * sin(ego_phi * np.pi / 180.)

        for other_idx in range(Para.MAX_OTHER_NUM):
            other = self.others[other_idx * Para.PER_OTHER_INFO_DIM:(other_idx + 1) * Para.PER_OTHER_INFO_DIM]
            other_x, other_y, other_phi, other_l, other_w = other[0], other[1], other[3], other[4], other[5]
            other_lws = (other_l - other_w) / 2.
            other_front_points = other_x + other_lws * math.cos(other_phi * np.pi / 180.), \
                                 other_y + other_lws * math.sin(other_phi * np.pi / 180.)
            other_rear_points = other_x - other_lws * math.cos(other_phi * np.pi / 180.), \
                                other_y - other_lws * math.sin(other_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for other_point in [other_front_points, other_rear_points]:
                    veh2veh_dist = \
                        sqrt(power(ego_point[0] - other_point[0], 2) + power(ego_point[1] - other_point[1], 2)) - 3.5
                    g_list.append(veh2veh_dist)
        if self.task == 'left':
            v_dist = if_else(ego_y < -Para.CROSSROAD_SIZE / 2, self.start_exp_v - ego_v,
                             if_else(ego_x < -Para.CROSSROAD_SIZE / 2, self.end_exp_v - ego_v,
                                     self.middle_exp_v - ego_v))
        elif self.task == 'straight':
            v_dist = if_else(ego_y < -Para.CROSSROAD_SIZE / 2, self.start_exp_v - ego_v,
                             if_else(ego_y > Para.CROSSROAD_SIZE / 2, self.end_exp_v - ego_v,
                                     self.middle_exp_v - ego_v))
        else:
            v_dist = if_else(ego_y < -Para.CROSSROAD_SIZE / 2, self.start_exp_v - ego_v,
                             if_else(ego_x > Para.CROSSROAD_SIZE / 2, self.end_exp_v - ego_v,
                                     self.middle_exp_v - ego_v))
        g_list.append(v_dist)
        # for ego_point in [ego_front_points]:
        #     g_list.append(if_else(logic_and(ego_point[1]<-18, ego_point[0]<1), ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[1]<-18, 3.75-ego_point[0]<1), 3.75-ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]>0, 0-ego_point[1]<0), 0-ego_point[1], 1))
        #     g_list.append(if_else(logic_and(ego_point[1]>-18, 3.75-ego_point[0]<1), 3.75-ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]<0, 7.5-ego_point[1]<1), 7.5-ego_point[1]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]<-18, ego_point[1]-0<1), ego_point[1]-0-1, 1))

        return g_list


class ModelPredictiveControl(object):
    def __init__(self, horizon, task, ref_index):
        self.horizon = horizon
        self.base_frequency = 10.
        self.task = task
        self.ref_index = ref_index
        self.DYNAMICS_DIM = Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM
        self.ACTION_DIM = 2
        self.dynamics = None
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init, XO):
        self.dynamics = Dynamics(x_init, self.task, self.ref_index, 1 / self.base_frequency)

        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # Create empty NLP
        w = []
        lbw = []  # lower bound for state and action constraints
        ubw = []  # upper bound for state and action constraints
        lbg = []  # lower bound for distance constraint
        ubg = []  # upper bound for distance constraint
        G = []  # dynamic constraints
        J = 0  # accumulated cost

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init[:self.DYNAMICS_DIM]
        ubw += x_init[:self.DYNAMICS_DIM]

        for k in range(1, self.horizon + 1):
            f = vertcat(*self.dynamics.f_xu(x, u))
            F = Function("F", [x, u], [f])
            g = vertcat(*self.dynamics.g_x(x))
            G_f = Function('Gf', [x], [g])

            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-0.3, -2.]
            ubw += [0.3, 1.]

            Fk = F(Xk, Uk)
            Gk = G_f(Xk)
            self.dynamics.other_pred()
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constraints
            G += [Fk - Xk]  # ego vehicle dynamic constraints
            lbg += [0.0] * self.DYNAMICS_DIM
            ubg += [0.0] * self.DYNAMICS_DIM
            G += [Gk]  # safety constraints
            lbg += [0.0] * (Para.MAX_OTHER_NUM * 4 + 1)
            ubg += [inf] * (Para.MAX_OTHER_NUM * 4 + 1)
            w += [Xk]
            lbw += [-inf] * self.DYNAMICS_DIM # speed constraints
            ubw += [inf] * self.DYNAMICS_DIM
            # Cost function TODO(guanyang): adjust all the parameters
            F_cost = Function('F_cost', [x, u], [0.15 * power(x[8], 2)
                                                 + 0.4 * power(x[6], 2)
                                                 + 20 * power(x[7] * np.pi / 180., 2)
                                                 + 0.02 * power(x[2], 2)
                                                 + 0.01 * power(x[1], 2)
                                                 + 5 * power(u[0], 2)
                                                 + 0.05 * power(u[1], 2)
                                                 ])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # load constraints and solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))
        state_all = np.array(r['x'])
        g_all = np.array(r['g'])
        state = np.zeros([self.horizon, self.DYNAMICS_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step
        cost = np.array(r['f']).squeeze(0)

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all, cost


class HierarchicalCompare:
    def __init__(self, exp_dir, ite, args):
        self.task = None
        self.policy = LoadPolicy(exp_dir, ite, args)
        self.horizon = 25
        self.env = IdcVirtualVehEnv()
        self.obs, _ = self.env.reset(green_prob=1.)
        self.DYNAMICS_DIM = Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM
        self.stg = MultiPathGenerator()
        self.mpc_cal_timer = TimerStat(window_size=1)
        self.adp_cal_timer = TimerStat(window_size=1)

    def run(self, is_render=False):
        self.obs, _ = self.env.reset(green_prob=1.)
        self.task = self.env.training_task
        path_list = self.stg.generate_path(self.task, 'green')
        info_of_a_run = []
        done = False
        while not done:
            print('step')
            info = dict(obs=self.obs)
            start_time = time.time()
            mpc_path_values, mpc_actions = [], []
            weight = [1.0, 1.0, 1.0]
            for ref_index, _ in enumerate(path_list):
                mpc = ModelPredictiveControl(self.horizon, self.task, ref_index)
                abso_obs = convert_to_abso(self.obs)
                state_all = np.array((list(abso_obs[:self.DYNAMICS_DIM]) + [0, 0]) * self.horizon +
                                     list(abso_obs[:self.DYNAMICS_DIM])).reshape((-1, 1))
                state, control, state_all, g_all, cost = mpc.mpc_solver(list(abso_obs), state_all)
                if any(g_all < -1):
                    print('optimization fail')
                    state_all = np.array((list(abso_obs[:self.DYNAMICS_DIM]) + [0, 0]) * self.horizon +
                                         list(abso_obs[:self.DYNAMICS_DIM])).reshape((-1, 1))
                    mpc_action = np.array([0., -2.])
                else:
                    state_all = np.array((list(abso_obs[:self.DYNAMICS_DIM]) + [0, 0]) * self.horizon +
                                         list(abso_obs[:self.DYNAMICS_DIM])).reshape((-1, 1))
                    mpc_action = control[0]
                mpc_path_values.append(weight[ref_index] * cost.squeeze().tolist())
                mpc_actions.append(mpc_action)
            mpc_path_values = np.array(mpc_path_values, dtype=np.float32)
            mpc_path_index = int(np.argmin(mpc_path_values))
            if self.obs[4] < -Para.CROSSROAD_SIZE / 2:
                mpc_path_index = 0
            mpc_action = mpc_actions[mpc_path_index]
            mpc_action = np.array([mpc_action[0] / Para.STEER_SCALE,
                                   (mpc_action[1] - Para.ACC_SHIFT) / Para.ACC_SCALE])  # in norm
            info.update(dict(mpc_path_values=mpc_path_values,
                             mpc_path_index=mpc_path_index,
                             mpc_action=mpc_action,
                             mpc_time=(time.time()-start_time)*1000))

            start_time = time.time()
            obs_list, mask_list = [], []
            other_vector, mask = self.env._construct_other_vector_short()
            ego_vector = self.env._construct_ego_vector_short()
            self.light_encoding = LIGHT_ENCODING[self.env.light_phase]
            for path in path_list:
                self.env.set_traj(path)
                track_vector = path.tracking_error_vector_vectorized(ego_vector[3], ego_vector[4], ego_vector[5], ego_vector[0])
                vector = np.concatenate((ego_vector, track_vector, self.light_encoding, self.env.task_encoding,
                                         path.ref_encoding, self.env.action_store[0], self.env.action_store[1],
                                         other_vector), axis=0)
                vector = vector.astype(np.float32)
                obs = convert_to_rela(vector)
                obs_list.append(obs)
                mask_list.append(mask)
            all_obs, all_mask = tf.stack(obs_list, axis=0), tf.stack(mask_list, axis=0)
            adp_path_values = self.policy.obj_value_batch(all_obs, all_mask).numpy()
            adp_actions, _ = self.policy.run_batch(all_obs, all_mask)
            adp_actions = adp_actions.numpy()
            adp_path_index = int(np.argmax(adp_path_values))
            if self.obs[4] < -Para.CROSSROAD_SIZE / 2:
                adp_path_index = 0
            adp_action = adp_actions[adp_path_index]  # in norm
            info.update(dict(adp_path_values=adp_path_values,
                             adp_path_index=adp_path_index,
                             adp_action=adp_action,
                             adp_time=(time.time()-start_time)*1000))
            self.env.set_traj(path_list[mpc_path_index])
            self.obs, rew, done, _ = self.env.step(mpc_action)
            info.update(dict(done=done,
                             done_type=self.env.done_type))
            info_of_a_run.append(copy.deepcopy(info))
            if is_render:
                self.render(path_list[mpc_path_index], mpc_path_values)
        return info_of_a_run

    def render(self, mpc_ref_path, mpc_values):
        render(light_phase=self.env.light_phase, all_other=self.env.all_other, detected_other=None,
               interested_other=self.env.interested_other, attn_weights=None,
               obs=self.obs, ref_path=mpc_ref_path, future_n_point=None, action=self.env.action,
               done_type=None, reward_info=self.env.reward_dict, hist_posi=None, path_values=mpc_values,
               sensor_config=None, is_debug=False)
        plt.show()
        plt.pause(0.1)


def compare_path_selection(exp_dir, ite, args, parallel_worker=4):
    ray.init()
    total_steps = 500
    steps = 0
    all_runs = []
    while steps < total_steps:
        alltask = [ray.remote(HierarchicalCompare).options(num_cpus=1).remote(exp_dir, ite, args) for _ in range(parallel_worker)]
        runs = ray.get([task.run.remote() for task in alltask])
        steps += sum([len(run) for run in runs])
        all_runs.extend(runs)
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data_and_plot/all_runs_{}.npy'.format(time_now)
    np.save(save_dir, all_runs)


def compare_from_same_start(exp_dir, ite, args, parallel_worker=4):
    ray.init()

    @ray.remote
    class OneTimeShoot:
        def __init__(self, alg, exp_dir, ite, args):
            self.env = IdcVirtualVehEnv()
            self.ref_index = self.env.ref_path.path_index
            self.task = self.env.training_task
            self.alg = alg
            self.DYNAMICS_DIM = Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM
            if alg == 'MPC':
                self.horizon = 25
                self.mpc_agent = ModelPredictiveControl(self.horizon, self.task, self.ref_index)
            else:
                self.rl_agent = LoadPolicy(exp_dir, ite, args)

        def run(self):
            self.obs, _ = self.env.reset(green_prob=1.)
            info_of_a_run = []
            if self.alg == 'MPC':
                for i in range(50):
                    abso_obs = convert_to_abso(self.obs)
                    state_all = np.array((list(abso_obs[:self.DYNAMICS_DIM]) + [0, 0]) * self.horizon +
                                         list(abso_obs[:self.DYNAMICS_DIM])).reshape((-1, 1))
                    state, control, state_all, g_all, cost = self.mpc_agent.mpc_solver(list(abso_obs), state_all)
                    if any(g_all < -1):
                        print('optimization fail')
                        state_all = np.array((list(abso_obs[:self.DYNAMICS_DIM]) + [0, 0]) * self.horizon +
                                             list(abso_obs[:self.DYNAMICS_DIM])).reshape((-1, 1))
                        mpc_action = np.array([0., -2.])
                    else:
                        state_all = np.array((list(abso_obs[:self.DYNAMICS_DIM]) + [0, 0]) * self.horizon +
                                             list(abso_obs[:self.DYNAMICS_DIM])).reshape((-1, 1))
                        mpc_action = control[0]
                    self.obs, rew, done, info = self.env.step(np.array([mpc_action[0]/Para.STEER_SCALE,
                                                                        (mpc_action[1]-Para.ACC_SHIFT)/Para.ACC_SCALE]))
                    reward_dict = info['reward_dict']
                    reward_dict.update(dict(is_done=done, done_type=self.env.done_type))
                    info_of_a_run.append(reward_dict)
            else:
                for i in range(50):
                    obs, mask, future_n_point = self.env._get_obs()
                    rl_action, _ = self.rl_agent.run_batch(obs[np.newaxis, :], mask[np.newaxis, :])
                    rl_action = rl_action[0]
                    self.obs, rew, done, info = self.env.step(rl_action)
                    reward_dict = info['reward_dict']
                    reward_dict.update(dict(is_done=done, done_type=self.env.done_type))
                    info_of_a_run.append(reward_dict)
            return info_of_a_run

    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    info_of_mpc_runs = []
    for i in range(math.ceil(2 / parallel_worker)):
        print('mpc: the {}-th parallel run ({} runs in total)'.format(i+1, i*parallel_worker))
        alltask = [OneTimeShoot.remote('MPC', exp_dir, ite, args) for j in range(parallel_worker)]
        info_of_mpc_runs.extend(ray.get([task.run.remote() for task in alltask]))
    save_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/data_and_plot/info_from_same_start_mpc_{}.npy'.format(time_now)
    np.save(save_dir, info_of_mpc_runs)
    info_of_rl_runs = []
    for i in range(math.ceil(2 / parallel_worker)):
        print('rl: the {}-th parallel run ({} runs in total)'.format(i+1, i*parallel_worker))
        alltask = [OneTimeShoot.remote('RL', exp_dir, ite, args) for j in range(parallel_worker)]
        info_of_rl_runs.extend(ray.get([task.run.remote() for task in alltask]))
    save_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/data_and_plot/info_from_same_start_rl_{}.npy'.format(time_now)
    np.save(save_dir, info_of_rl_runs)


def main():
    exp_dir = RESULTS_DIR + '/ampc/data2plot/experiment-2022-05-17-13-35-51'
    ite = 195000
    parser = argparse.ArgumentParser()
    params = json.loads(open(exp_dir + '/config.json').read())
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args = parser.parse_args()
    hier_comp = HierarchicalCompare(exp_dir, ite, args)
    plt.ion()
    # f = plt.figure(figsize=(pt2inch(420 / 4), pt2inch(420 / 4)), dpi=300)
    for i in range(100):
        hier_comp.run(is_render=True)
    # plt.close(f)


def exp():
    exp_dir = RESULTS_DIR + '/ampc/data2plot/experiment-2022-05-17-13-35-51'
    ite = 195000
    args = get_args(exp_dir)
    parallel_worker = 10
    # compare_from_same_start(exp_dir, ite, args, parallel_worker)
    compare_path_selection(exp_dir, ite, args, parallel_worker)


if __name__ == '__main__':
    # main()
    exp()
