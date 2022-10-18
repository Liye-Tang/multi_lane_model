#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import numpy as np
from algorithm.preprocessor import Preprocessor
from algorithm.utils.misc import TimerStat
from algorithm.policy import PolicyWithQs
from env_and_model import Name2EnvAndModelCls


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.replay_batch_size
        self.policy_with_value = PolicyWithQs(**vars(self.args))
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_for_policy_update = self.args.num_rollout_for_policy_update
        _, model_cls = Name2EnvAndModelCls[self.args.env_id]
        self.model = model_cls()
        self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
                                         rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
                                         args=self.args)
        self.grad_timer = TimerStat()
        self.stats = {}

    def get_stats(self):
        return self.stats

    def get_batch_data(self, batch_data):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def model_rollout_for_policy_update(self, start_obses):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        punish_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses
        reward_dict_list = []
        for _ in range(self.num_rollout_for_policy_update):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses, reward_dict = self.model.rollout(actions)
            reward_dict_list.append(reward_dict)
            rewards_sum += self.preprocessor.tf_process_rewards(reward_dict['rewards'])
            punish_sum += self.preprocessor.tf_process_punish(reward_dict['punish'])

        policy_loss = -self.tf.reduce_mean(rewards_sum)
        info = dict(policy_loss=policy_loss)
        for k, _ in reward_dict_list[0].items():
            k_sum = self.tf.reduce_sum([reward_dict[k] for reward_dict in reward_dict_list], axis=0)
            info[k + '_sum_model'] = self.tf.reduce_mean(k_sum)
            info[k + '_mean_model'] = self.tf.reduce_mean([reward_dict[k] for reward_dict in reward_dict_list])
        return info

    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape() as tape:
            info = self.model_rollout_for_policy_update(mb_obs)

        with self.tf.name_scope('policy_gradient') as scope:
            policy_grad = tape.gradient(info['policy_loss'], self.policy_with_value.policy.trainable_weights)
            policy_grad, policy_grad_norm = self.tf.clip_by_global_norm(policy_grad, self.args.gradient_clip_norm)
            info.update(dict(policy_grad_norm=policy_grad_norm))
            return policy_grad, info

    def compute_gradient(self, samples, iteration):
        self.get_batch_data(samples)
        mb_obs = self.batch_data['batch_obs']
        with self.grad_timer:
            policy_grad, info = self.policy_forward_and_backward(mb_obs)
        info = {k: v.numpy() for k, v in info.items()}
        self.stats.update(info)
        self.stats.update(dict(iteration=iteration,
                               grad_time=self.grad_timer.mean,))
        gradient_tensor = policy_grad
        return list(map(lambda x: x.numpy(), gradient_tensor)), self.get_stats()


class AMPCLearnerWithAttention(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.replay_batch_size
        self.policy_with_value = PolicyWithQs(**vars(self.args))
        self.batch_data = {}
        self.all_data = {}
        self.num_rollout_for_policy_update = self.args.num_rollout_for_policy_update
        _, model_cls = Name2EnvAndModelCls[self.args.env_id]
        self.model = model_cls()
        self.preprocessor = Preprocessor(obs_scale=self.args.obs_scale, rew_scale=self.args.rew_scale,
                                         rew_shift=self.args.rew_shift, punish_scale=self.args.punish_scale,
                                         args=self.args)
        self.grad_timer = TimerStat()
        self.stats = {}

    def get_stats(self):
        return self.stats
    
    def get_states(self, mb_obs, mb_mask):
        mb_obs_others, mb_attn_weights = self.policy_with_value.compute_attn(mb_obs[:, self.args.other_start_dim:], mb_mask)
        mb_state = self.tf.concat((mb_obs[:, :self.args.other_start_dim], mb_obs_others), axis=1)
        return mb_state

    def get_batch_data(self, batch_data):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_future_n_point': batch_data[-2].astype(np.float32),
                           'batch_mask': batch_data[-1].astype(np.float32),
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite//interval, self.tf.float32))
        return pf
    
    def model_rollout_for_update(self, mb_obs, ite, mb_future_n_point, mb_mask):
        self.model.reset(mb_obs, mb_future_n_point)
        rewards_sum = self.tf.zeros((self.batch_size,))
        punish_sum = self.tf.zeros((self.batch_size,))
        rewards4value_sum = self.tf.zeros((self.batch_size,))

        pf = self.punish_factor_schedule(ite)
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        mb_state = self.get_states(processed_mb_obs, mb_mask)
        actions, _ = self.policy_with_value.compute_action(mb_state)
        pred_q = self.policy_with_value.compute_Qs(self.tf.stop_gradient(mb_state),
                                                   self.tf.stop_gradient(actions))[0]  # use the first one
        reward_dict_list = []
        for i in range(self.num_rollout_for_policy_update):
            processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
            mb_state = self.get_states(processed_mb_obs, mb_mask)
            actions, _ = self.policy_with_value.compute_action(mb_state)
            mb_obs, reward_dict = self.model.rollout(actions)  # mb_future_n_point [#batch, 4, T]
            reward_dict_list.append(reward_dict)
            rewards_sum += self.preprocessor.tf_process_rewards(reward_dict['rewards'])
            punish_sum += self.preprocessor.tf_process_punish(reward_dict['punish'])
            rewards4value_sum += self.preprocessor.tf_process_rewards(reward_dict['rewards4value'])
        # obj v loss
        q_loss = self.tf.reduce_mean(self.tf.square(pred_q - self.tf.stop_gradient(rewards4value_sum)))

        # pg loss
        obj_loss = -self.tf.reduce_mean(rewards_sum)
        policy_loss = obj_loss
        con_loss = self.tf.reduce_mean(punish_sum)
        con_loss_with_pf = self.tf.stop_gradient(pf) * con_loss
        policy_loss += con_loss_with_pf
        info = dict(policy_loss=policy_loss,
                    obj_loss=obj_loss,
                    con_loss=con_loss,
                    q_loss=q_loss,
                    con_loss_with_pf=con_loss_with_pf,
                    pf=pf)
        for k, _ in reward_dict_list[0].items():
            k_sum = self.tf.reduce_sum([reward_dict[k] for reward_dict in reward_dict_list], axis=0)
            info[k + '_sum_model'] = self.tf.reduce_mean(k_sum)
            info[k + '_mean_model'] = self.tf.reduce_mean([reward_dict[k] for reward_dict in reward_dict_list])
        return info

    @tf.function
    def forward_and_backward(self, mb_obs, ite, mb_future_n_point, mb_mask):
        with self.tf.GradientTape(persistent=True) as tape:
            info = self.model_rollout_for_update(mb_obs, ite, mb_future_n_point, mb_mask)
        policy_grad = tape.gradient(info['policy_loss'], self.policy_with_value.policy.trainable_weights)
        attn_net_grad = tape.gradient(info['policy_loss'], self.policy_with_value.attn_net.trainable_weights)
        q_grad = tape.gradient(info['q_loss'], self.policy_with_value.Qs[0].trainable_weights)
        policy_grad, policy_grad_norm = self.tf.clip_by_global_norm(policy_grad, self.args.gradient_clip_norm)
        q_grad, q_grad_norm = self.tf.clip_by_global_norm(q_grad, self.args.gradient_clip_norm)
        attn_net_grad, attn_net_grad_norm = self.tf.clip_by_global_norm(attn_net_grad, self.args.gradient_clip_norm)
        info.update(dict(policy_grad_norm=policy_grad_norm,
                         q_grad_norm=q_grad_norm,
                         attn_net_grad_norm=attn_net_grad_norm))
        grad_list = [q_grad, policy_grad, attn_net_grad]
        return grad_list, info

    def compute_gradient(self, samples, iteration):
        self.get_batch_data(samples)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        mb_future_n_point = self.tf.constant(self.batch_data['batch_future_n_point'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        mb_mask = self.tf.constant(self.batch_data['batch_mask'])
        with self.grad_timer:
            grad_list, info = self.forward_and_backward(mb_obs, iteration, mb_future_n_point, mb_mask)
        info = {k: v.numpy() for k, v in info.items()}
        self.stats.update(info)
        self.stats.update(dict(iteration=iteration,
                               grad_time=self.grad_timer.mean,))
        gradient_tensor = sum(grad_list, [])
        return list(map(lambda x: x.numpy(), gradient_tensor)), self.get_stats()


if __name__ == '__main__':
    pass
