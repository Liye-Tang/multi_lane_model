#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging
import os
from collections import deque

import numpy as np

from algorithm.policy import PolicyWithQs, PolicyWithVs
from algorithm.preprocessor import Preprocessor
from algorithm.utils.misc import judge_is_nan, safemean, TimerStat
from algorithm.utils.monitor import Monitor, MonitorMultiAgent
from env_and_model import Name2EnvAndModelCls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OffPolicyWorkerWithAttention(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        env_cls, _ = Name2EnvAndModelCls[self.args.env_id]
        self.env = env_cls()
        self.policy_with_value = PolicyWithQs(**vars(self.args))
        self.sample_batch_size = self.args.sample_batch_size
        self.obs, self.info = self.env.reset()
        self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
                                         rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
                                         args=self.args)
        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.stats = {}
        logger.info('Worker initialized')
    
    def get_stats(self):
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
                               )
                          )
        return self.stats

    def set_stats(self, stats):
        self.stats = stats
    
    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        self.policy_with_value.apply_gradients(self.tf.constant(iteration, dtype=self.tf.int32), grads)

    def _get_state(self, obs, mask):
        obs_other, _ = self.policy_with_value.compute_attn(obs[self.args.other_start_dim:][np.newaxis, :],
                                                           mask[np.newaxis, :])
        obs_other = obs_other.numpy()[0]
        state = np.concatenate((obs[:self.args.other_start_dim], obs_other), axis=0)
        return state

    def sample(self):
        batch_data = []
        reward_dict_list = []
        for _ in range(self.sample_batch_size):
            processed_obs = self.preprocessor.process_obs(self.obs)
            mask = self.info['mask']
            state = self._get_state(processed_obs, mask)
            action, _ = self.policy_with_value.compute_action(state[np.newaxis, :])
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            # try:
            #     judge_is_nan([action])
            # except ValueError:
            #     print('processed_obs', processed_obs)
            #     print('policy_weights', self.policy_with_value.policy.trainable_weights)
            #     action, _ = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
            #     judge_is_nan([action])
            #     raise ValueError
            obs_tp1, reward, done, info = self.env.step(action.numpy()[0])
            reward_dict_list.append(info['reward_dict'])
            punish, reward4value = info['reward_dict']['punish'], info['reward_dict']['rewards4value']
            batch_data.append((self.obs.copy(), action.numpy()[0], reward, obs_tp1.copy(),
                               done, punish, reward4value, self.info['future_n_point'], self.info['mask']))
            if done:
                self.obs, self.info = self.env.reset()
            else:
                self.obs = obs_tp1.copy()
                self.info = info.copy()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))
        for k, _ in reward_dict_list[0].items():
            self.stats[k+'_mean_data'] = self.tf.reduce_mean([reward_dict[k] for reward_dict in reward_dict_list])
        self.num_sample += len(batch_data)
        self.sample_times += 1
        return batch_data

    def sample_with_stats(self):
        batch_data = self.sample()
        return batch_data, self.get_stats()


class OffPolicyWorker(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        self.num_agent = self.args.num_agent
        env_cls, _ = Name2EnvAndModelCls[self.args.env_id]
        self.env = env_cls()
        self.policy_with_value = PolicyWithQs(**vars(self.args))
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
                                         rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
                                         args=self.args)
        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.stats = {}
        logger.info('Worker initialized')

    def get_stats(self):
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
                               )
                          )
        return self.stats

    def set_stats(self, stats):
        self.stats = stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        # self.policy_with_value.apply_gradients(self.tf.constant(iteration, dtype=self.tf.int32), grads)
        self.policy_with_value.apply_gradients(iteration, grads)

    def sample(self):
        batch_data = []
        reward_dict_list = []
        for _ in range(int(self.sample_batch_size / self.num_agent)):
            reward_dict_list.append({})
            for i in range(self.num_agent):
                batch_data.append(
                    (self.obs.copy(), 0, 0, 0, 0, 0)
                )
            self.obs = self.env.reset()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))
        # for k, _ in reward_dict_list[0].items():
        #     self.stats[k+'_mean_data'] = self.tf.reduce_mean([reward_dict[k] for reward_dict in reward_dict_list])
        self.num_sample += len(batch_data)
        self.sample_times += 1
        return batch_data

    def sample_with_stats(self):
        batch_data = self.sample()
        return batch_data, self.get_stats()


Name2WorkerCls = dict(offpolicy=OffPolicyWorker,
                      offpolicy_with_attn=OffPolicyWorkerWithAttention)


