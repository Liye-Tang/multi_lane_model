#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/5/15
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: e2e_planner.py
# =====================================
import argparse
import json
import time
import ray

import tensorflow as tf

from algorithm.policy import PolicyWithQs
from algorithm.preprocessor import Preprocessor
from env_and_model.idc_virtual.e2e_planner.end2end import E2eEnv
from env_and_model.idc_virtual.endtoend_env_utils import *


class LoadPolicy(object):
    def __init__(self, exp_dir, iter, args):
        model_dir = exp_dir + '/models'
        self.args = args
        self.policy = PolicyWithQs(**vars(self.args))
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor(self.args.obs_scale, self.args.rew_scale,
                                         self.args.rew_shift, self.args.punish_scale,
                                         self.args)
        for i in [1, 2, 3]:
            init_obs = np.zeros((i, self.args.obs_dim))
            self.run_batch(init_obs)

    @tf.function
    def run_batch(self, mb_obs):
        processed_mb_obs = self.preprocessor.np_process_obses(mb_obs)
        actions, _ = self.policy.compute_action(processed_mb_obs)
        return actions


class E2ePlanner:
    def __init__(self, exp_dir, ite, args):
        self.env = E2eEnv(mode='testing')
        self.policy = LoadPolicy(exp_dir, ite, args)

    def run_an_episode(self, is_render=False):
        obs, _ = self.env.reset()
        done = False
        episode_info = []
        while not done:
            action = self.policy.run_batch(obs[np.newaxis, :])[0]
            start_time = time.time()
            obs, rew, done, info = self.env.step(action)
            comp_info = {'cal_time_ms': (time.time() - start_time) * 1000,
                         'a_x': action[1] * Para.ACC_SCALE + Para.ACC_SHIFT,
                         'pass_time_s': 0.1,
                         'done': done,
                         'done_type': self.env.done_type}
            episode_info.append(comp_info)
            if is_render:
                self.env.render()
        return episode_info


def main():
    exp_dir = ''
    ite = ''
    e2e_planner = E2ePlanner(exp_dir, ite)
    for i in range(20):
        e2e_planner.run_an_episode(is_render=True)


if __name__ == '__main__':
    main()
