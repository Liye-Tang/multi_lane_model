#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: preprocessor.py
# =====================================

import numpy as np
import tensorflow as tf


class Preprocessor(object):
    def __init__(self, obs_scale=None, rew_scale=None, rew_shift=None, punish_scale=None, args=None, **kwargs):
        self.obs_scale = np.array(obs_scale, dtype=np.float32) if obs_scale is not None else None
        self.rew_scale = rew_scale
        self.rew_shift = rew_shift
        self.punish_scale = punish_scale
        self.args = args

    def process_rew(self, rew):
        if self.rew_scale:
            return (rew + self.rew_shift) * self.rew_scale
        else:
            return rew

    def process_obs(self, obs):
        if self.obs_scale is not None:
            return obs * self.obs_scale
        else:
            return obs

    def process_punish(self, punish):
        if self.punish_scale:
            return punish * self.punish_scale
        else:
            return punish

    def np_process_obses(self, obses):
        if self.obs_scale is not None:
            return obses * self.obs_scale
        else:
            return obses

    def tf_process_obses(self, obses):
        if self.obs_scale is not None:
            return obses * tf.convert_to_tensor(self.obs_scale, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(obses, dtype=tf.float32)

    def np_process_rewards(self, rewards):
        if self.rew_scale:
            return (rewards + self.rew_shift) * self.rew_scale
        else:
            return rewards

    def tf_process_rewards(self, rewards):
        if self.rew_scale:
            return (rewards+tf.convert_to_tensor(self.rew_shift, dtype=tf.float32)) \
                   * tf.convert_to_tensor(self.rew_scale, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(rewards, dtype=tf.float32)

    def np_process_punish(self, punish):
        if self.punish_scale:
            return punish * self.punish_scale
        else:
            return punish

    def tf_process_punish(self, punish):
        if self.punish_scale:
            return punish * tf.convert_to_tensor(self.punish_scale, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(punish, dtype=tf.float32)
