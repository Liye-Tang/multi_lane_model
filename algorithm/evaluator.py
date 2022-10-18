#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import os

import numpy as np

from algorithm.preprocessor import Preprocessor
from algorithm.utils.misc import TimerStat
from env_and_model import Name2EnvAndModelCls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EvaluatorWithAttention(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        env_cls, _ = Name2EnvAndModelCls[self.args.env_id]
        self.env = env_cls()
        self.policy_with_value = policy_cls(**vars(self.args))
        self.iteration = 0
        if self.args.mode == 'training' or self.args.mode == 'debug':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
                                         rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
                                         args=self.args)
        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def evaluate_saved_model(self, model_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)

    def _get_state(self, obs, mask):
        obs_other, weights = self.policy_with_value.compute_attn(obs[self.args.other_start_dim:][np.newaxis, :],
                                                                 mask[np.newaxis, :])
        obs_other = obs_other.numpy()[0]
        weights = weights.numpy()[0]
        state = np.concatenate((obs[:self.args.other_start_dim], obs_other), axis=0)
        return state, weights

    def run_an_episode(self, steps=None, render=True):
        reward_list = []
        reward_dict_list = []
        done = 0
        obs, info = self.env.reset()
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.process_obs(obs)
                mask = info['mask']
                state, attn_weights = self._get_state(processed_obs, mask)
                action = self.policy_with_value.compute_mode(state[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                if render: self.env.render(attn_weights=attn_weights)
                reward_list.append(reward)
                reward_dict_list.append(info['reward_dict'])
        else:
            while not done:
                processed_obs = self.preprocessor.process_obs(obs)
                mask = info['mask']
                state, attn_weights = self._get_state(processed_obs, mask)
                action = self.policy_with_value.compute_mode(state[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                if render: self.env.render(attn_weights=attn_weights)
                reward_list.append(reward)
                reward_dict_list.append(info['reward_dict'])

        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in reward_dict_list[0].keys():
            info_key = list(map(lambda x: x[key], reward_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key + '_mean': mean_key,
                              key + '_sum': sum(info_key)})
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n):
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            n_info_dict = self.run_n_episode(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), n_info_dict))
        self.eval_times += 1

    def get_eval_times(self):
        return self.eval_times


# class EvaluatorWithAttention(object):
#     import tensorflow as tf
#     tf.config.experimental.set_visible_devices([], 'GPU')
#     tf.config.threading.set_inter_op_parallelism_threads(1)
#     tf.config.threading.set_intra_op_parallelism_threads(1)
#
#     def __init__(self, policy_cls, args):
#         logging.getLogger("tensorflow").setLevel(logging.ERROR)
#         self.args = args
#         _, model_cls = Name2EnvAndModelCls[self.args.env_id]
#         self.fake_env = model_cls(use_model_as_env=True)
#         self.policy_with_value = policy_cls(**vars(self.args))
#         self.iteration = 0
#         if self.args.mode == 'training' or self.args.mode == 'debug':
#             self.log_dir = self.args.log_dir + '/evaluator'
#         else:
#             self.log_dir = self.args.test_log_dir
#         if not os.path.exists(self.log_dir):
#             os.makedirs(self.log_dir)
#
#         self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
#                                          rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
#                                          args=self.args)
#         self.writer = self.tf.summary.create_file_writer(self.log_dir)
#         self.stats = {}
#         self.eval_timer = TimerStat()
#         self.eval_times = 0
#
#     def get_stats(self):
#         self.stats.update(dict(eval_time=self.eval_timer.mean))
#         return self.stats
#
#     def load_weights(self, load_dir, iteration):
#         self.policy_with_value.load_weights(load_dir, iteration)
#
#     def evaluate_saved_model(self, model_load_dir, iteration):
#         self.load_weights(model_load_dir, iteration)
#
#     def _get_state(self, obs, mask):
#         obs_other, weights = self.policy_with_value.compute_attn(obs[self.args.other_start_dim:][np.newaxis, :],
#                                                                  mask[np.newaxis, :])
#         obs_other = obs_other.numpy()[0]
#         weights = weights.numpy()[0]
#         state = np.concatenate((obs[:self.args.other_start_dim], obs_other), axis=0)
#         return state, weights
#
#     def run_an_episode(self, steps=None, render=True):
#         reward_list = []
#         reward_dict_list = []
#         done = 0
#         obs, info = self.env.reset()
#         if render: self.env.render()
#         if steps is not None:
#             for _ in range(steps):
#                 processed_obs = self.preprocessor.process_obs(obs)
#                 mask = info['mask']
#                 state, attn_weights = self._get_state(processed_obs, mask)
#                 action = self.policy_with_value.compute_mode(state[np.newaxis, :])
#                 obs, reward, done, info = self.env.step(action.numpy()[0])
#                 if render: self.env.render(attn_weights=attn_weights)
#                 reward_list.append(reward)
#                 reward_dict_list.append(info['reward_dict'])
#         else:
#             while not done:
#                 processed_obs = self.preprocessor.process_obs(obs)
#                 mask = info['mask']
#                 state, attn_weights = self._get_state(processed_obs, mask)
#                 action = self.policy_with_value.compute_mode(state[np.newaxis, :])
#                 obs, reward, done, info = self.env.step(action.numpy()[0])
#                 if render: self.env.render(attn_weights=attn_weights)
#                 reward_list.append(reward)
#                 reward_dict_list.append(info['reward_dict'])
#
#         episode_return = sum(reward_list)
#         episode_len = len(reward_list)
#         info_dict = dict()
#         for key in reward_dict_list[0].keys():
#             info_key = list(map(lambda x: x[key], reward_dict_list))
#             mean_key = sum(info_key) / len(info_key)
#             info_dict.update({key + '_mean': mean_key,
#                               key + '_sum': sum(info_key)})
#         info_dict.update(dict(episode_return=episode_return,
#                               episode_len=episode_len))
#         return info_dict
#
#     def run_n_episode(self, n):
#         list_of_info_dict = []
#         for _ in range(n):
#             logger.info('logging {}-th episode'.format(_))
#             info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
#             list_of_info_dict.append(info_dict.copy())
#         n_info_dict = dict()
#         for key in list_of_info_dict[0].keys():
#             info_key = list(map(lambda x: x[key], list_of_info_dict))
#             mean_key = sum(info_key) / len(info_key)
#             n_info_dict.update({key: mean_key})
#         return n_info_dict
#
#     def set_weights(self, weights):
#         self.policy_with_value.set_weights(weights)
#
#     def run_evaluation(self, iteration):
#         with self.eval_timer:
#             self.iteration = iteration
#             n_info_dict = self.run_n_episode(self.args.num_eval_episode)
#             with self.writer.as_default():
#                 for key, val in n_info_dict.items():
#                     self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
#                 for key, val in self.get_stats().items():
#                     self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
#                 self.writer.flush()
#         if self.eval_times % self.args.eval_log_interval == 0:
#             logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), n_info_dict))
#         self.eval_times += 1
#
#     def get_eval_times(self):
#         return self.eval_times

class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        env_cls, _ = Name2EnvAndModelCls[self.args.env_id]
        self.env = env_cls(num_agent=self.args.num_eval_agent)
        self.policy_with_value = policy_cls(**vars(self.args))
        self.iteration = 0
        if self.args.mode == 'training' or self.args.mode == 'debug':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
                                         rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
                                         args=self.args)
        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def evaluate_saved_model(self, model_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)

    def run_an_episode(self, steps=None, render=True):
        reward_list = []
        reward_dict_list = []
        done = 0
        obs = self.env.reset(total=True)
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                obs, reward, done, info = self.env.step(action.numpy())
                if render: self.env.render()
                reward_list.append(reward[0])
                reward_dict_list.append(info['reward_dict'])
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                obs, reward, done, info = self.env.step(action.numpy())
                done = done[0]
                if render: self.env.render()
                reward_list.append(reward[0])
                reward_dict_list.append(info['reward_dict'])
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in reward_dict_list[0].keys():
            info_key = list(map(lambda x: x[key][0], reward_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key + '_mean': mean_key,
                              key + '_sum': sum(info_key)})
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episodes(self, n):
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def run_n_episodes_parallel(self, n):
        logger.info('logging {} episodes in parallel'.format(n))
        rewards_list = []
        reward_dict_list = []
        obses = self.env.reset(total=True)
        if self.args.eval_render: self.env.render()
        for _ in range(self.args.fixed_steps):
            processed_obs = self.preprocessor.tf_process_obses(obses)
            processed_obses = self.tf.expand_dims(processed_obs, 0)
            actions = self.policy_with_value.compute_mode(processed_obses)
            obses, rewards, dones, info = self.env.step(actions[0].numpy())
            if self.args.eval_render: self.env.render()
            rewards_list.append(rewards)
            reward_dict_list.append(info['reward_dict'])

        list_of_info_dict = []
        for i in range(n):
            reward_list_i = [rewards[i] for rewards in rewards_list]
            episode_return_i = sum(reward_list_i)
            episode_len_i = len(reward_list_i)
            info_dict_i = dict()
            for key in reward_dict_list[0].keys():
                info_key = list(map(lambda x: x[key][i], reward_dict_list))
                mean_key = sum(info_key) / len(info_key)
                info_dict_i.update({key + '_mean': mean_key,
                                    key + '_sum': sum(info_key)})
            info_dict_i.update(dict(episode_return=episode_return_i,
                                    episode_len=episode_len_i))
            list_of_info_dict.append(info_dict_i)
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            if self.args.num_eval_agent == 1:
                n_info_dict = self.run_n_episodes(self.args.num_eval_episode)
            else:
                n_info_dict = self.run_n_episodes_parallel(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), n_info_dict))
        self.eval_times += 1


Name2EvaluatorCls = dict(evaluator=Evaluator,
                         evaluator_with_attn=EvaluatorWithAttention)


if __name__ == '__main__':
    pass
