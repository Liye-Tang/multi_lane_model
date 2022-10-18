#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import json
import shutil

import tensorflow as tf

from env_and_model.idc_real.dynamics_and_models import EnvironmentModel, ReferencePath
from env_and_model.idc_real.endtoend import CrossroadEnd2endMix
from env_and_model.idc_real.endtoend_env_utils import *
from env_and_model.idc_real.hierarchical_decision.multi_path_generator import MultiPathGenerator
from env_and_model.idc_real.utils.load_policy import LoadPolicy
from env_and_model.idc_real.utils.misc import TimerStat
from env_and_model.idc_real.utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, train_exp_dir, ite, logdir=None):
        self.policy = LoadPolicy('../utils/models/{}'.format(train_exp_dir), ite)
        self.args = self.policy.args
        # TODO(guanyang): determine the future point num
        self.env = CrossroadEnd2endMix(mode='testing', future_point_num=self.args.num_rollout_list_for_policy_update[0])
        self.model = EnvironmentModel(mode='testing')
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = MultiPathGenerator()
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.fig_plot = 0
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

    def reset(self, ):
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
                                                                   self.episode_counter - 1),
                                                               isshow=False)
        return self.obs

    @tf.function
    def is_safe(self, obses, masks, future_n_point):
        self.model.reset(obses, future_n_point)
        punish = 0.
        # TODO(guanyang): determine the rollout length
        for step in range(20):
            action, _ = self.policy.run_batch(obses, masks)
            obses, reward_dict = self.model.rollout(action)
            punish += reward_dict['veh2veh4real'][0] + reward_dict['veh2bike4real'][0] + \
                      reward_dict['veh2person4real'][0]
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

    @tf.function
    def cal_prediction_obs(self, real_obs, real_mask, real_future_n_point):
        obses, masks = real_obs[np.newaxis, :], real_mask[np.newaxis, :]
        ref_points = tf.expand_dims(real_future_n_point, axis=0)
        self.model.reset(obses)
        obses_list = []
        for i in range(25):
            action, _ = self.policy.run_batch(obses, masks)
            obses, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
                veh2person4real, _, reward_dict = self.model.rollout_online(action, ref_points[:, :, i])
            obses_list.append(obses[0])
        return tf.convert_to_tensor(obses_list)

    def step(self):
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

        self.render(self.path_list[path_index], weights, path_values)
        self.recorder.record(obs_real, safe_action, self.step_timer.mean,
                             path_index, path_values, self.ss_timer.mean,
                             is_ss)
        self.obs, r, done, info = self.env.step(safe_action)
        return done

    def action_postprocess(self, obs, action):
        curr_v, _, _, ego_x, ego_y, _ = \
            obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
        norm_steer, norm_acc = action[0], action[1]
        acc = norm_acc * Para.ACC_SCALE + Para.ACC_SHIFT
        dt = 0.1
        max_speed = 8
        if acc < 0 and curr_v + acc * dt < 0:
            acc = -curr_v / dt
        elif acc > 0 and curr_v + acc * dt > max_speed:
            acc = (max_speed - curr_v) / dt
        norm_acc = (acc + 0.5) / 1.5
        return np.array([norm_steer, norm_acc], dtype=np.float32)

    def render(self, ref_path, attn_weights, path_values):
        render(light_phase=self.env.light_phase, all_other=self.env.all_other,
               interested_other=self.env.interested_other, attn_weights=attn_weights, obs=self.obs, ref_path=ref_path,
               future_n_point=None, action=self.env.action, done_type=self.env.done_type,
               reward_info=self.env.reward_info, hist_posi=self.hist_posi, path_values=path_values)
        plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(
                self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.pdf'.format(self.step_counter))


def plot_and_save_ith_episode_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    save_dir = logdir + '/episode{}/figs'.format(i)
    os.makedirs(save_dir, exist_ok=True)
    recorder.plot_and_save_ith_episode_curves(i, save_dir, True)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    hier_decision = HierarchicalDecision('experiment-2021-12-14-20-02-54', 150000, logdir)

    for i in range(300):
        for _ in range(500):
            done = hier_decision.step()
            if done:
                print(hier_decision.env.done_type)
                break
        hier_decision.reset()


def plot_static_path():
    ax = render(light_phase=None, all_other=None, interested_other=None,
                attn_weights=None, obs=None, ref_path=None, future_n_point=None, action=None,
                done_type=None, reward_info=None, hist_posi=None, path_values=None)

    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list
        control_points = path.control_points
        color = Para.PATH_COLOR

        for i, (path_x, path_y, _) in enumerate(path_list):
            ax.plot(path_x[600:-600], path_y[600:-600], color=color[i])
        for i, four_points in enumerate(control_points):
            for point in four_points:
                ax.scatter(point[0], point[1], color=color[i], s=20, alpha=0.7)
            ax.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--',
                    color=color[i], alpha=0.5)
            ax.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--',
                    color=color[i], alpha=0.5)
            ax.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--',
                    color=color[i], alpha=0.5)

    plt.savefig('./multipath_planning.pdf')
    plt.show()


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    interval = file_num // (num-1)
    start = file_num % (num-1)
    # print(start, file_num, interval)
    selected = [start//2] + [start//2+interval*i for i in range(1, num-1)]
    # print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{}.jpg'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.jpg'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_and_save_ith_episode_data('./results/good/2021-03-15-23-56-21', 0)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)
