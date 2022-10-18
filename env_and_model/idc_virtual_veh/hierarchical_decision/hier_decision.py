#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import json
import os.path

import tensorflow as tf

from env_and_model.idc_virtual_veh.dynamics_and_models import IdcVirtualVehModel
from env_and_model.idc_virtual_veh.endtoend import IdcVirtualVehEnv
from env_and_model.idc_virtual_veh.endtoend_env_utils import *
from env_and_model.idc_virtual_veh.hierarchical_decision.multi_path_generator import MultiPathGenerator
from env_and_model.idc_virtual_veh.utils.load_policy import LoadPolicy, get_args
from env_and_model.idc_virtual_veh.utils.misc import TimerStat
from env_and_model.idc_virtual_veh.utils.recorder import Recorder
from utils import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class HierarchicalDecision(object):
    def __init__(self, exp_dir, ite, args, logdir=None):
        self.policy = LoadPolicy(exp_dir, ite, args)
        self.args = args
        self.env = IdcVirtualVehEnv(mode='testing')
        self.model = IdcVirtualVehModel()
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = MultiPathGenerator()
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(exp_dir=exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.hist_posi = []
        self.old_index = 0
        self.path_list = self.stg.generate_path(self.env.training_task,
                                                LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        # ------------------build graph for tf.function in advance-----------------------
        obs, all_info = self.env.reset()
        mask, future_n_point = all_info['mask'], all_info['future_n_point']
        obs = tf.convert_to_tensor(obs[np.newaxis, :], dtype=tf.float32)
        mask = tf.convert_to_tensor(mask[np.newaxis, :], dtype=tf.float32)
        future_n_point = tf.convert_to_tensor(future_n_point[np.newaxis, :], dtype=tf.float32)
        self.is_safe(obs, mask, future_n_point)
        self.policy.run_batch(obs, mask)
        self.policy.obj_value_batch(obs, mask)
        # ------------------build graph for tf.function in advance-----------------------
        self.reset()

    def reset(self,):
        self.obs, _ = self.env.reset()
        self.path_list = self.stg.generate_path(self.env.training_task,
                                                LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        self.recorder.reset()
        self.old_index = 0
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.step_counter = -1
            self.recorder.save(self.logdir)
            if self.episode_counter >= 1:
                select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter - 1, 12)
                self.recorder.plot_and_save_ith_episode_curves(self.episode_counter - 1,
                                                               self.logdir + '/episode{}/figs'.format(
                                                                   self.episode_counter - 1))
        return self.obs

    @tf.function
    def is_safe(self, obses, masks, future_n_point):
        self.model.reset(obses, future_n_point)
        punish = 0.
        # TODO(guanyang): determine the rollout length
        for step in range(5):
            action, _ = self.policy.run_batch(obses, masks)
            obses, reward_dict = self.model.rollout(action)
            punish += reward_dict['real_punish_term'][0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs, real_mask, real_future_n_point):
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :], dtype=tf.float32)
        real_mask = tf.convert_to_tensor(real_mask[np.newaxis, :], dtype=tf.float32)
        real_future_n_point = tf.convert_to_tensor(real_future_n_point[np.newaxis, :], dtype=tf.float32)
        if not self.is_safe(real_obs, real_mask, real_future_n_point):
            print('SAFETY SHIELD STARTED!')
            _, weight = self.policy.run_batch(real_obs, real_mask)
            return np.array([0.0, -1.0], dtype=np.float32), weight.numpy()[0], True
        else:
            action, weight = self.policy.run_batch(real_obs, real_mask)
            action, weight = action.numpy()[0], weight.numpy()[0]
            # TODO(guanyang): add post cmd process
            return action, weight, False

    def step(self, is_render=True):
        self.step_counter += 1
        self.path_list = self.stg.generate_path(self.env.training_task,
                                                LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        with self.step_timer:
            obs_list, mask_list, future_n_point_list = [], [], []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                obs, mask, future_n_point = self.env._get_obs()
                obs_list.append(obs)
                mask_list.append(mask)
                future_n_point_list.append(future_n_point)
            all_obs, all_mask = tf.stack(obs_list, axis=0), tf.stack(mask_list, axis=0)
            path_values = self.policy.obj_value_batch(all_obs, all_mask).numpy()
            path_values = np.array([path_values[i] if self.old_index - 1 <= i <= self.old_index + 1 else -10000
                                    for i in range(len(path_values))], dtype=np.float32)
            old_value = path_values[self.old_index]
            # value is to approximate (- sum of reward)
            new_index, new_value = np.argmax(path_values), max(path_values)
            # TODO(guanyang): determine the frequency and the threshold
            if self.step_counter % 10 == 0:
                path_index = self.old_index if old_value - new_value < 0.01 else new_index
            else:
                path_index = self.old_index
            self.old_index = path_index
            self.env.set_traj(self.path_list[path_index])
            obs_real, mask_real, future_n_point_real = \
                obs_list[path_index], mask_list[path_index], future_n_point_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, weights, is_ss = self.safe_shield(obs_real, mask_real, future_n_point_real)
            safe_action = self.action_postprocess(obs_real, safe_action)

        if is_render:
            self.render(self.path_list[path_index], weights, path_values)
        self.recorder.record(obs_real, safe_action, self.step_timer.mean,
                             path_index, path_values, self.ss_timer.mean,
                             is_ss)
        self.obs, r, done, info = self.env.step(safe_action)
        comp_info = {'cal_time_ms': (self.step_timer.mean-self.ss_timer.mean) * 1000,
                     'total_time_ms': self.step_timer.mean * 1000,
                     'ss_time_ms': self.ss_timer.mean * 1000,
                     'a_x': safe_action[0] * Para.ACC_SCALE + Para.ACC_SHIFT,
                     'pass_time_s': 0.1, 'done': done, 'done_type': self.env.done_type}
        return done, comp_info

    def action_postprocess(self, obs, action):
        curr_v, _, _, ego_x, ego_y, _ = \
            obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
        norm_steer, norm_acc = action[0], action[1]
        acc = norm_acc * Para.ACC_SCALE + Para.ACC_SHIFT
        dt = 0.1
        max_speed = 6.66
        if acc < 0 and curr_v + acc * dt < 0:
            acc = -curr_v / dt
        elif acc > 0 and curr_v + acc * dt > max_speed:
            acc = (max_speed - curr_v) / dt
        norm_acc = (acc - Para.ACC_SHIFT) / Para.ACC_SCALE
        return np.array([norm_steer, norm_acc], dtype=np.float32)

    def render(self, ref_path, attn_weights, path_values):
        render(light_phase=self.env.light_phase, all_other=self.env.all_other, detected_other=None,
               interested_other=self.env.interested_other, attn_weights=attn_weights, obs=self.obs, ref_path=ref_path,
               future_n_point=None, action=self.env.action, done_type=self.env.done_type,
               reward_info=self.env.reward_dict, hist_posi=self.hist_posi, path_values=path_values,
               sensor_config=[(80., 360.), (100., 38.)], is_debug=False)
        plt.show()
        plt.pause(0.05)
        if self.logdir is not None:
            plt.savefig(
                self.logdir + '/episode{}'.format(self.episode_counter) + '/step{:03d}.pdf'.format(self.step_counter))

    def run_an_episode(self, is_render=False):
        done = False
        episode_info = []
        f = plt.figure(figsize=(pt2inch(420 / 4), pt2inch(420 / 4)), dpi=300)
        plt.ion()
        while not done:
            done, comp_info = self.step(is_render=is_render)
            episode_info.append(comp_info)
        plt.close(f)
        self.reset()
        return episode_info


def plot_and_save_ith_episode_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    save_dir = logdir + '/episode{}/figs'.format(i)
    os.makedirs(save_dir, exist_ok=True)
    recorder.plot_and_save_ith_episode_curves(i, save_dir)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.dirname(CURRENT_DIR) + '/data_and_plot/single_demo/{time}'.format(time=time_now)
    os.makedirs(logdir)
    exp_dir = RESULTS_DIR + '/ampc/data2plot/experiment-2022-05-13-16-36-06'
    ite = 198000
    args = get_args(exp_dir)
    hier_decision = HierarchicalDecision(exp_dir, ite, args, logdir)
    for i in range(300):
        hier_decision.run_an_episode(is_render=True)


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    intervavl = file_num // (num - 1)
    start = file_num % (num - 1)
    print(start, file_num, intervavl)
    selected = [start // 2] + [start // 2 + intervavl * i - 1 for i in range(1, num)]
    print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{:03d}.pdf'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.pdf'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_and_save_ith_episode_data('./results/good/2021-03-15-23-56-21', 0)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)
