#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import numpy as np
import tensorflow as tf

from algorithm.policy import PolicyWithQs
from algorithm.preprocessor import Preprocessor
from env_and_model.idc_virtual_veh.endtoend_env_utils import *
from utils import RESULTS_DIR

def get_args(exp_dir):
    parser = argparse.ArgumentParser()
    params = json.loads(open(exp_dir + '/config.json').read())
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    return parser.parse_args()


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
            init_obs, init_mask = np.zeros((i, self.args.obs_dim), np.float32),\
                                  np.zeros((i, Para.MAX_OTHER_NUM), np.float32)
            self.run_batch(init_obs, init_mask)
            self.obj_value_batch(init_obs, init_mask)

    @tf.function
    def run_batch(self, mb_obs, mb_mask):
        processed_mb_obs = self.preprocessor.np_process_obses(mb_obs)
        states, mb_attn_weights = self.get_states(processed_mb_obs, mb_mask)
        actions, _ = self.policy.compute_action(states)
        return actions, mb_attn_weights

    @tf.function
    def obj_value_batch(self, mb_obs, mb_mask):
        processed_mb_obs = self.preprocessor.np_process_obses(mb_obs)
        states, _ = self.get_states(processed_mb_obs, mb_mask)
        actions, _ = self.policy.compute_action(states)
        values = self.policy.compute_Qs(states, actions)[-1]
        return values

    def get_states(self, mb_obs, mb_mask):
        mb_obs_others, mb_attn_weights = self.policy.compute_attn(mb_obs[:, self.args.other_start_dim:], mb_mask)
        mb_state = tf.concat((mb_obs[:, :self.args.other_start_dim], mb_obs_others), axis=1)
        return mb_state, mb_attn_weights


def policy_test():
    policy_dir = RESULTS_DIR + '/ampc/data2plot/experiment-2022-05-13-16-36-06'
    ite = 198000
    policy = LoadPolicy(policy_dir, ite)
    print(policy.run_batch(np.zeros((1, policy.args.obs_dim), np.float32),
                           np.zeros((1, Para.MAX_OTHER_NUM), np.float32)))


if __name__ == '__main__':
    policy_test()

