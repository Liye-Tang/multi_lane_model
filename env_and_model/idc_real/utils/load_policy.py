#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import tensorflow as tf

from algorithm.policy import AttentionPolicy4Toyota
from algorithm.preprocessor import Preprocessor
from env_and_model.idc_real.endtoend_env_utils import *

# TODO: add LoadPolicy to compatible with several learners
class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir + '/models'
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        self.policy = AttentionPolicy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor(self.args.obs_scale, self.args.rew_scale,
                                         self.args.rew_shift, self.args.punish_scale,
                                         self.args)
        init_obs, init_mask = np.zeros((1, self.args.obs_dim)), np.zeros((1, Para.MAX_OTHER_NUM))
        init_obs_with_specific_shape, init_mask_with_specific_shape =\
            np.tile(init_obs, (3, 1)), np.tile(init_mask, (3, 1)),
        self.run_batch(init_obs, init_mask)
        self.obj_value_batch(init_obs, init_mask)
        self.run_batch(init_obs_with_specific_shape, init_mask_with_specific_shape)
        self.obj_value_batch(init_obs_with_specific_shape, init_mask_with_specific_shape)

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
        values = self.policy.compute_obj_v(states)
        return values

    def get_states(self, mb_obs, mb_mask):
        mb_obs_others, mb_attn_weights = self.policy.compute_attn(mb_obs[:, self.args.other_start_dim:], mb_mask)
        mb_state = tf.concat((mb_obs[:, :self.args.other_start_dim], mb_obs_others), axis=1)
        return mb_state, mb_attn_weights
