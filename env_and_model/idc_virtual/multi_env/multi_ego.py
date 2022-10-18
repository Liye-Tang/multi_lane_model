#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: multi_ego.py
# =====================================

import copy
import datetime
import time

import tensorflow as tf
from matplotlib.patches import Wedge

from env_and_model.idc_virtual.dynamics_and_models import IdcVirtualModel
from env_and_model.idc_virtual.endtoend import IdcVirtualEnv
from env_and_model.idc_virtual.endtoend_env_utils import *
from env_and_model.idc_virtual.hierarchical_decision.hier_decision import \
    select_and_rename_snapshots_of_an_episode
from env_and_model.idc_virtual.hierarchical_decision.multi_path_generator import MultiPathGenerator
from env_and_model.idc_virtual.traffic import Traffic
from env_and_model.idc_virtual.utils.load_policy import LoadPolicy
from utils import *

NAME2TASK = dict(DL='left', DU='straight', DR='right',
                 RD='left', RL='straight', RU='right',
                 UR='left', UD='straight', UL='right',
                 LU='left', LR='straight', LD='right')
ROTATE_ANGLE = dict(D=0, R=90, U=180, L=-90)
dirname = os.path.dirname(__file__)


class MultiEgo(object):
    def __init__(self, init_n_ego_dict, exp_dir, ite):  # init_n_ego_dict is used to init traffic (mainly) and ego dynamics
        self.policy = LoadPolicy(exp_dir, ite)
        self.n_ego_instance = {}
        self.n_ego_dynamics = {}
        self.n_ego_select_index = {}
        for egoID, ego_dict in init_n_ego_dict.items():
            self.n_ego_instance[egoID] = IdcVirtualEnv(mode='testing', multi_display=True)
        self.mpp = MultiPathGenerator()
        self.model = IdcVirtualModel()
        self.egoID2pop = []

        # ------------------build graph for tf.function in advance-----------------------
        fake_obses = np.zeros([1, Para.OBS_DIM], dtype=np.float32)
        fake_masks = np.zeros([1, Para.MAX_OTHER_NUM], dtype=np.float32)
        fake_future_n_points = np.zeros([1, 4, Para.FUTURE_POINT_NUM], dtype=np.float32)
        self.is_safe(fake_obses, fake_masks, fake_future_n_points)
        # ------------------build graph for tf.function in advance-----------------------
        self.reset(init_n_ego_dict)

    def reset(self, init_n_ego_dict):
        self.egoID2pop = []
        for egoID, ego_dict in init_n_ego_dict.items():
            self.n_ego_dynamics[egoID] = \
                self.n_ego_instance[egoID]._get_ego_dynamics(
                    [ego_dict['v_x'], ego_dict['v_y'], ego_dict['r'], ego_dict['x'], ego_dict['y'], ego_dict['phi']],
                    [0, 0, self.n_ego_instance[egoID].dynamics.vehicle_params['miu'], self.n_ego_instance[egoID].dynamics.vehicle_params['miu']])

    def get_traffic_light(self, egoID, v_light):
        rotate_angle = ROTATE_ANGLE[egoID[0]]
        if rotate_angle == 0 or rotate_angle == 180:
            v_light_trans = v_light
        else:
            v_light_trans = 0 if v_light != 3 and v_light != 4 else 2
        return v_light_trans

    def get_next_n_ego_dynamics(self, n_ego_others, v_light):
        for egoID, ego_dynamics in self.n_ego_dynamics.items():
            rotate_angle = ROTATE_ANGLE[egoID[0]]
            others = n_ego_others[egoID]
            others_trans = cal_info_in_transform_coordination(others, 0, 0, rotate_angle)
            ego_dynamics_trans = cal_ego_info_in_transform_coordination(ego_dynamics, 0, 0, rotate_angle)
            v_light_trans = self.get_traffic_light(egoID, v_light)
            self.n_ego_instance[egoID].all_other = others_trans
            self.n_ego_instance[egoID].ego_dynamics = ego_dynamics_trans
            self.n_ego_instance[egoID].light_phase = v_light_trans
            # generate multiple paths
            task = NAME2TASK[egoID[:2]]
            path_list = self.mpp.generate_path(task, LIGHT_PHASE_TO_GREEN_OR_RED[v_light_trans])
            obs_list, mask_list, future_n_point_list = [], [], []
            # evaluate each path
            for path in path_list:
                self.n_ego_instance[egoID].set_traj(path)
                obs, other_mask, future_n_point = self.n_ego_instance[egoID]._get_obs(exit_=egoID[0])
                obs_list.append(obs)
                mask_list.append(other_mask)
                future_n_point_list.append(future_n_point)
            all_obs = tf.stack(obs_list, axis=0)
            all_mask = tf.stack(mask_list, axis=0)
            path_values = self.policy.obj_value_batch(all_obs, all_mask).numpy()
            # select and safety shield
            path_index = int(np.argmax(path_values))
            self.n_ego_select_index[egoID] = path_index
            obs_real, mask_real, future_n_point_real = \
                obs_list[path_index], mask_list[path_index], future_n_point_list[path_index]
            # safe shield
            safe_action = self.safe_shield(obs_real[np.newaxis, :], mask_real[np.newaxis, :],
                                           future_n_point_real[np.newaxis, :], egoID)
            action_trans = self.n_ego_instance[egoID]._action_transformation_for_end2end(safe_action)
            next_ego_state, next_ego_params = self.n_ego_instance[egoID]._get_next_ego_state(action_trans)
            next_ego_dynamics = self.n_ego_instance[egoID]._get_ego_dynamics(next_ego_state, next_ego_params)
            self.n_ego_dynamics[egoID] = cal_ego_info_in_transform_coordination(next_ego_dynamics, 0, 0, -rotate_angle)
        return copy.deepcopy(self.n_ego_dynamics)

    def judge_n_ego_done(self, n_ego_collision_flag):
        n_ego_done = {}
        for egoID in self.n_ego_dynamics.keys():
            ego_instance = self.n_ego_instance[egoID]
            collision_flag = n_ego_collision_flag[egoID]
            is_achieve_goal = ego_instance._is_achieve_goal()
            n_ego_done[egoID] = [collision_flag, is_achieve_goal]
        return n_ego_done

    @tf.function
    def is_safe(self, obses, masks, future_n_points):
        self.model.reset(obses, future_n_points)
        punish = 0.
        # TODO(guanyang): determine the rollout steps
        for step in range(20):
            actions = self.policy.run_batch(obses, masks)
            obses, reward_dict = self.model.rollout(actions)
            punish += reward_dict['real_punish_term'][0]
        return False if punish > 0 else True

    def safe_shield(self, obses, masks, future_n_points, egoID):
        if not self.is_safe(obses, future_n_points):
            print(egoID + ': SAFETY SHIELD STARTED!')
            return np.array([0., -1.], dtype=np.float32)
        else:
            return self.policy.run_batch(obses, masks).numpy().squeeze(0)


class Simulation(object):
    def __init__(self, init_n_ego_dict, exp_dir, ite, logdir):
        self.init_n_ego_dict = init_n_ego_dict
        self.multiego = MultiEgo(copy.deepcopy(self.init_n_ego_dict), exp_dir, ite)
        self.traffic = Traffic(100, 'display', self.init_n_ego_dict)
        self.n_ego_traj_trans = {}
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.hist_posi = {egoID: [] for egoID in self.init_n_ego_dict.keys()}
        self.episode_counter = -1
        self.step_counter = -1
        self.logdir = logdir
        self.reset()

    def reset(self):
        self.multiego.reset(copy.deepcopy(self.init_n_ego_dict))
        self.traffic.init_traffic(copy.deepcopy(self.init_n_ego_dict))
        self.traffic.sim_step()
        v_light = self.traffic.light_phase
        n_ego_traj = {egoID: self.multiego.mpp.generate_path(NAME2TASK[egoID[:2]],
                                                             self.multiego.get_traffic_light(egoID, v_light))
                      for egoID in self.multiego.n_ego_dynamics.keys()}
        self.n_ego_traj_trans = {}
        for egoID, ego_traj in n_ego_traj.items():
            traj_list = []
            for item in ego_traj:
                temp = np.array([rotate_coordination(x, y, 0, -ROTATE_ANGLE[egoID[0]])[0] for x, y in
                                 zip(item.path[0], item.path[1])]), \
                       np.array([rotate_coordination(x, y, 0, -ROTATE_ANGLE[egoID[0]])[1] for x, y in
                                 zip(item.path[0], item.path[1])])
                traj_list.append(temp)
            self.n_ego_traj_trans[egoID] = traj_list
        self.hist_posi = {egoID: [] for egoID in self.init_n_ego_dict.keys()}
        if self.logdir is not None:
            self.episode_counter += 1
            self.step_counter = -1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            if self.episode_counter >= 1:
                select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter - 1, 12)

    def step(self):
        self.step_counter += 1
        current_n_ego_others = self.traffic.n_ego_others
        current_v_light = self.traffic.light_phase
        current_n_ego_collision_flag = self.traffic.n_ego_collision_flag
        next_n_ego_dynamics = self.multiego.get_next_n_ego_dynamics(current_n_ego_others, current_v_light)
        n_ego_done = self.multiego.judge_n_ego_done(current_n_ego_collision_flag)
        for egoID, flag_list in n_ego_done.items():
            if flag_list[0]:
                print('Ego {} collision!'.format(egoID))
                return 1
            elif flag_list[1]:
                print('Ego {} achieve goal!'.format(egoID))
                self.multiego.n_ego_dynamics.pop(egoID)
                self.traffic.n_ego_dict.pop(egoID)
                next_n_ego_dynamics.pop(egoID)

                self.multiego.n_ego_select_index.pop(egoID)
                self.n_ego_traj_trans.pop(egoID)
                if len(self.traffic.n_ego_dict) == 0:
                    print('All ego achieve goal!'.format(egoID))

                    return 1
        self.traffic.set_own_car(next_n_ego_dynamics)
        self.traffic.sim_step()
        return 0

    def render(self,):
        n_ego_vehicles = {egoID: self.multiego.n_ego_instance[egoID].all_other for egoID in
                          self.multiego.n_ego_dynamics.keys()}
        n_ego_dynamics = {egoID: self.multiego.n_ego_instance[egoID].ego_dynamics for egoID in
                          self.multiego.n_ego_dynamics.keys()}
        some_egoID = list(n_ego_vehicles.keys())[0]
        all_other = cal_info_in_transform_coordination(n_ego_vehicles[some_egoID], 0, 0,
                                                       -ROTATE_ANGLE[some_egoID[0]])
        ax = render(light_phase=self.traffic.light_phase, all_other=all_other, detected_other=None,
                    interested_other=None, attn_weights=None,
                    obs=None, ref_path=None, future_n_point=None, action=None,
                    done_type=None, reward_info=None, hist_posi=None, path_values=None,
                    sensor_config=None)

        def draw_rec(x, y, phi, l, w, facecolor, edgecolor, alpha=1.):
            phi_rad = phi * np.pi / 180
            bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -phi)
            ax.add_patch(
                plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=edgecolor, linewidth=linewidth,
                              facecolor=facecolor, angle=-(90 - phi), alpha=alpha, zorder=50))
            ax.plot([x, x + (l / 2 + 1) * np.cos(phi_rad)], [y, y + (l / 2 + 1) * np.sin(phi_rad)], linewidth=linewidth,
                    color=edgecolor)

        n_ego_dynamics_trans = {}
        for egoID, ego_dynamics in n_ego_dynamics.items():
            n_ego_dynamics_trans[egoID] = cal_ego_info_in_transform_coordination(ego_dynamics, 0, 0,
                                                                                 -ROTATE_ANGLE[egoID[0]])
        # plot own car
        for egoID, ego_info in n_ego_dynamics_trans.items():
            ego_x = ego_info['x']
            ego_y = ego_info['y']
            ego_a = ego_info['phi']
            ego_l = ego_info['l']
            ego_w = ego_info['w']
            self.hist_posi[egoID].append((ego_x, ego_y))
            draw_rec(ego_x, ego_y, ego_a, ego_l, ego_w, facecolor='fuchsia', edgecolor='k')
        # plot history
        xs, ys = [], []
        for egoID in self.init_n_ego_dict.keys():
            xs += [pos[0] for pos in self.hist_posi[egoID]]
            ys += [pos[1] for pos in self.hist_posi[egoID]]
        ax.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.1)

        # plot history
        xs, ys, ts = [], [], []
        for egoID in self.init_n_ego_dict.keys():
            freq = 5
            xs += [pos[0] for i, pos in enumerate(self.hist_posi[egoID]) if i % freq == 0]
            ys += [pos[1] for i, pos in enumerate(self.hist_posi[egoID]) if i % freq == 0]
            ts += [0.1 * i for i, pos in enumerate(self.hist_posi[egoID]) if i % freq == 0]
            ax.scatter(np.array(xs), np.array(ys), marker='o', c=ts, cmap='Wistia', alpha=0.1, s=0.8, zorder=40)

        # plot trajectory
        light_phase = self.traffic.light_phase
        if light_phase == 0 or light_phase == 1:
            v_color, h_color = 'green', 'red'
        elif light_phase == 2:
            v_color, h_color = 'orange', 'red'
        elif light_phase == 3 or light_phase == 4:
            v_color, h_color = 'red', 'green'
        else:
            v_color, h_color = 'red', 'orange'
        for egoID, planed_traj in self.n_ego_traj_trans.items():
            for i, path in enumerate(planed_traj):
                alpha = 1
                if v_color != 'green':
                    if egoID[:2] in ['DL', 'DU', 'UD', 'UR']:
                        alpha = 0.2
                if h_color != 'green':
                    if egoID[:2] in ['RD', 'RL', 'LR', 'LU']:
                        alpha = 0.2
                if planed_traj is not None:
                    if i == self.multiego.n_ego_select_index[egoID]:
                        ax.plot(path[0], path[1], color=Para.PATH_COLOR[i], alpha=alpha)
                    else:
                        ax.plot(path[0], path[1], color=Para.PATH_COLOR[i], alpha=0.2)
        # plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{:03d}.pdf'.format(self.step_counter))


def main():
    init_n_ego_dict = dict(
        DL1=dict(v_x=5, v_y=0, r=0, x=0.5 * Para.LANE_WIDTH, y=-30, phi=90, l=Para.L, w=Para.W, routeID='dl'),
        DU1=dict(v_x=8, v_y=0, r=0, x=1.5 * Para.LANE_WIDTH, y=-45, phi=90, l=Para.L, w=Para.W, routeID='du'),
        DR1=dict(v_x=5, v_y=0, r=0, x=2.5 * Para.LANE_WIDTH, y=-30, phi=90, l=Para.L, w=Para.W, routeID='dr'),
        RD1=dict(v_x=3, v_y=0, r=0, x=31.5, y=0.5 * Para.LANE_WIDTH, phi=180, l=Para.L, w=Para.W, routeID='rd'),
        RL1=dict(v_x=5, v_y=0, r=0, x=33, y=1.5 * Para.LANE_WIDTH, phi=180, l=Para.L, w=Para.W, routeID='rl'),
        RU1=dict(v_x=5, v_y=0, r=0, x=38, y=2.5 * Para.LANE_WIDTH, phi=180, l=Para.L, w=Para.W, routeID='ru'),
        UR1=dict(v_x=5, v_y=0, r=0, x=-0.5 * Para.LANE_WIDTH, y=32, phi=-90, l=Para.L, w=Para.W, routeID='ur'),
        UD1=dict(v_x=5, v_y=0, r=0, x=-1.5 * Para.LANE_WIDTH, y=50, phi=-90, l=Para.L, w=Para.W, routeID='ud'),
        UL1=dict(v_x=5, v_y=0, r=0, x=-2.5 * Para.LANE_WIDTH, y=50, phi=-90, l=Para.L, w=Para.W, routeID='ul'),
        LU1=dict(v_x=5, v_y=0, r=0, x=-34, y=-0.5 * Para.LANE_WIDTH, phi=0, l=Para.L, w=Para.W, routeID='lu'),
        LR1=dict(v_x=5, v_y=0, r=0, x=-32, y=-1.5 * Para.LANE_WIDTH, phi=0, l=Para.L, w=Para.W, routeID='lr'),
        LD1=dict(v_x=5, v_y=0, r=0, x=-30, y=-2.5 * Para.LANE_WIDTH, phi=0, l=Para.L, w=Para.W, routeID='ld'),
    )
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.dirname(dirname) + '/data_and_plot/multi_demo/{time}'.format(time=time_now)
    os.makedirs(logdir)
    simulation = Simulation(init_n_ego_dict, RESULTS_DIR + '/mpg2/experiment-2021-06-05-13-15-31', 300000, logdir)
    done = 0
    while 1:
        f = plt.figure(figsize=(pt2inch(420 / 4), pt2inch(420 / 4)), dpi=300)
        while not done:
            done = simulation.step()
            if not done:
                start_time = time.time()
                simulation.render()
                end_time = time.time()
                print('render time:', end_time -start_time)
        plt.close(f)
        simulation.reset()
        print('NEW EPISODE*********************************')
        done = 0


if __name__ == '__main__':
    main()
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-16-14-35-17', 0, 12)

