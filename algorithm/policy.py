#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from algorithm.model import Name2ModelCls, AttentionNet, AlphaModel, PINet


class AttentionPolicy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.state_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls, attn_model_cls = Name2ModelCls[self.args.value_model_cls], \
                                                            Name2ModelCls[self.args.policy_model_cls], \
                                                            Name2ModelCls[self.args.attn_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v',
                                     output_activation='softplus')
        obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

        # add AttentionNet
        attn_in_total_dim, attn_in_per_dim, attn_out_dim = self.args.attn_in_total_dim, \
                                                           self.args.attn_in_per_dim, \
                                                           self.args.attn_out_dim
        self.attn_net = attn_model_cls(attn_in_total_dim, attn_in_per_dim, attn_out_dim, name='attn_net')
        attn_lr_schedule = PolynomialDecay(*self.args.attn_lr_schedule)
        self.attn_optimizer = self.tf.keras.optimizers.Adam(attn_lr_schedule, name='adam_opt_attn')

        self.models = (self.obj_v, self.policy, self.attn_net)
        self.optimizers = (self.obj_value_optimizer, self.policy_optimizer, self.attn_optimizer)
    
    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        obj_v_len = len(self.obj_v.trainable_weights)
        pg_len = len(self.policy.trainable_weights)
        obj_v_grad, policy_grad = grads[:obj_v_len], grads[obj_v_len:obj_v_len+pg_len]
        attn_grad = grads[obj_v_len + pg_len:]
        self.obj_value_optimizer.apply_gradients(zip(obj_v_grad, self.obj_v.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        self.attn_optimizer.apply_gradients(zip(attn_grad, self.attn_net.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_obj_v(self, obs):
        with self.tf.name_scope('compute_obj_v') as scope:
            return tf.squeeze(self.obj_v(obs), axis=1)

    @tf.function
    def compute_attn(self, obs_others, mask):
        with self.tf.name_scope('compute_attn') as scope:
            return self.attn_net([obs_others, mask]) # return (logits, weights) tuple


class PolicyWithQs(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, obs_dim, act_dim,
                 value_model_cls, value_num_hidden_layers, value_num_hidden_units,
                 value_hidden_activation, value_out_activation, value_lr_schedule,
                 policy_model_cls, policy_num_hidden_layers, policy_num_hidden_units, policy_hidden_activation,
                 policy_out_activation, policy_lr_schedule,
                 alpha, alpha_lr_schedule, Q_num, target, tau, delay_update,
                 deterministic_policy, action_range, is_attn, state_dim, attn_lr_schedule,
                 attn_in_total_dim, attn_in_per_dim, attn_out_dim,
                 **kwargs):
        super().__init__()
        self.Q_num = Q_num
        self.target = target
        self.tau = tau
        self.delay_update = delay_update
        self.deterministic_policy = deterministic_policy
        self.action_range = action_range
        self.alpha = alpha
        self.is_attn = is_attn
        value_model_cls, policy_model_cls = Name2ModelCls[value_model_cls], Name2ModelCls[policy_model_cls]
        obs_or_state_dim = obs_dim if not is_attn else state_dim
        self.policy = policy_model_cls(obs_or_state_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                       policy_hidden_activation, act_dim * 2, name='policy',
                                       output_activation=policy_out_activation)
        self.policy_target = policy_model_cls(obs_or_state_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                              policy_hidden_activation, act_dim * 2, name='policy_target',
                                              output_activation=policy_out_activation)
        self.policy_target.set_weights(self.policy.get_weights())

        policy_lr = PolynomialDecay(*policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr, name='policy_adam_opt')
        self.Qs, self.Q_targets, self.Q_optimizers = [], [], []
        for i in range(self.Q_num):
            self.Qs.append(value_model_cls(obs_or_state_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                           value_hidden_activation, 1, name='Q' + str(i+1),
                                           output_activation=value_out_activation))
            self.Q_targets.append(value_model_cls(obs_or_state_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                                  value_hidden_activation, 1, name='Q' + str(i + 1) + '_target',
                                                  output_activation=value_out_activation),)
            self.Q_optimizers.append(self.tf.keras.optimizers.Adam(PolynomialDecay(*value_lr_schedule),
                                                                   name='Q'+str(i+1)+'_adam_opt'))
        if kwargs['learner_type'] == 'mpg2' or kwargs['learner_type'] == 'mpo_td3':
            for i in [0, 1, 4, 5]:
                if i+2 < self.Q_num:
                    self.Qs[i+2].set_weights(self.Qs[i].get_weights())
        for i in range(self.Q_num):
            self.Q_targets[i].set_weights(self.Qs[i].get_weights())

        self.models = self.Qs + [self.policy,]
        self.optimizers = self.Q_optimizers + [self.policy_optimizer,]
        self.target_models = self.Q_targets + [self.policy_target,] if self.target else ()

        if self.is_attn:
            # assert value_out_activation == 'softplus'
            if kwargs['attn_type'] == 'pi':
                print('attn_type is PINet')
                self.attn_net = PINet(attn_in_total_dim, attn_in_per_dim, attn_out_dim, name='attn_net')
            else:
                self.attn_net = AttentionNet(attn_in_total_dim, attn_in_per_dim, attn_out_dim, name='attn_net')
            attn_lr_schedule = PolynomialDecay(*attn_lr_schedule)
            self.attn_optimizer = self.tf.keras.optimizers.Adam(attn_lr_schedule, name='attn_adam_opt')
            self.models += (self.attn_net,)
            self.optimizers += (self.attn_optimizer,)
        if self.alpha == 'auto':
            self.alpha_model = AlphaModel(name='alpha')
            alpha_lr = self.tf.keras.optimizers.schedules.PolynomialDecay(*alpha_lr_schedule)
            self.alpha_optimizer = self.tf.keras.optimizers.Adam(alpha_lr, name='alpha_adam_opt')
            self.models += (self.alpha_model,)
            self.optimizers += (self.alpha_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models] + \
               [model.get_weights() for model in self.target_models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            if i < len(self.models):
                self.models[i].set_weights(weight)
            else:
                self.target_models[i-len(self.models)].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        q_weights_len = len(self.Qs[0].trainable_weights) if self.Q_num > 0 else 0
        policy_weights_len = len(self.policy.trainable_weights)
        for i in range(self.Q_num):
            self.Q_optimizers[i].apply_gradients(zip(grads[i*q_weights_len:(i+1)*q_weights_len],
                                                     self.Qs[i].trainable_weights))
        if iteration % self.delay_update == 0:
            policy_grad = grads[self.Q_num * q_weights_len:self.Q_num * q_weights_len + policy_weights_len]
            self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
            if self.is_attn:
                attn_start_dim = self.Q_num * q_weights_len + policy_weights_len
                attn_weights_len = len(self.attn_net.trainable_weights)
                attn_grad = grads[attn_start_dim:attn_start_dim+attn_weights_len]
                self.attn_optimizer.apply_gradients(zip(attn_grad, self.attn_net.trainable_weights))
            if self.target:
                self.update_policy_target()
                self.update_Q_targets()
            if self.alpha == 'auto':
                alpha_grad = grads[-1:]
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))

    def update_Q_targets(self):
        assert len(self.Qs) == len(self.Q_targets), 'check whether the target is set to True'
        tau = self.tau
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            for source, target in zip(Q.trainable_weights, Q_target.trainable_weights):
                target.assign(tau * source + (1.0 - tau) * target)

    def update_policy_target(self):
        tau = self.tau
        for source, target in zip(self.policy.trainable_weights, self.policy_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        # TODO(guanyang): check if need to clip
        log_std = tf.clip_by_value(log_std, -5., -1.)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.deterministic_policy:
                mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_target_action(self, obs):
        with self.tf.name_scope('compute_target_action') as scope:
            logits = self.policy_target(obs)
            if self.deterministic_policy:
                mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_Qs(self, obs, act):
        with self.tf.name_scope('compute_Qs') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return [self.tf.squeeze(Q(Q_inputs), axis=1) for Q in self.Qs]

    @tf.function
    def compute_Q_targets(self, obs, act):
        with self.tf.name_scope('compute_Q_targets') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return [self.tf.squeeze(Q_target(Q_inputs), axis=1) for Q_target in self.Q_targets]

    @property
    def log_alpha(self):
        return self.alpha_model.log_alpha

    @tf.function
    def compute_attn(self, obs_others, mask):
        with self.tf.name_scope('compute_attn') as scope:
            return self.attn_net([obs_others, mask])  # return (logits, weights) tuple


class PolicyWithVs(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, obs_dim, act_dim,
                 value_model_cls, value_num_hidden_layers, value_num_hidden_units,
                 value_hidden_activation, value_out_activation, value_lr_schedule,
                 policy_model_cls, policy_num_hidden_layers, policy_num_hidden_units, policy_hidden_activation,
                 policy_out_activation, policy_lr_schedule, V_num, action_range, is_attn, state_dim, attn_lr_schedule,
                 attn_in_total_dim, attn_in_per_dim, attn_out_dim, **kwargs):
        super().__init__()
        self.V_num = V_num
        self.is_attn = is_attn
        self.action_range = action_range
        value_model_cls, policy_model_cls = Name2ModelCls[value_model_cls], Name2ModelCls[policy_model_cls]
        obs_or_state_dim = obs_dim if not is_attn else state_dim
        self.policy = policy_model_cls(obs_or_state_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                       policy_hidden_activation, act_dim * 2, name='policy',
                                       output_activation=policy_out_activation)
        policy_lr = self.tf.keras.optimizers.schedules.PolynomialDecay(*policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr)
        self.Vs = tuple(value_model_cls(obs_or_state_dim, value_num_hidden_layers, value_num_hidden_units,
                                        value_hidden_activation, 1, name='V' + str(i + 1),
                                        output_activation=value_out_activation)
                        for i in range(self.V_num))
        value_lr = self.tf.keras.optimizers.schedules.PolynomialDecay(*value_lr_schedule)
        self.V_optimizers = tuple(self.tf.keras.optimizers.Adam(value_lr, name='V' + str(i + 1) + '_adam_opt')
                                      for i in range(len(self.Vs)))
        self.models = self.Vs + (self.policy,)
        self.optimizers = self.V_optimizers + (self.policy_optimizer,)

        if self.is_attn:
            assert value_out_activation == 'softplus'
            self.attn_net = AttentionNet(attn_in_total_dim, attn_in_per_dim, attn_out_dim, name='attn_net')
            attn_lr_schedule = PolynomialDecay(*attn_lr_schedule)
            self.attn_optimizer = self.tf.keras.optimizers.Adam(attn_lr_schedule, name='attn_adam_opt')
            self.models += (self.attn_net,)
            self.optimizers += (self.attn_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_grads_sepe(self, grads):
        v_weights_len = len(self.Vs[0].trainable_weights) if self.V_num > 0 else 0
        policy_weights_len = len(self.policy.trainable_weights)
        for i in range(self.V_num):
            self.V_optimizers[i].apply_gradients(zip(grads[i * v_weights_len:(i + 1) * v_weights_len],
                                                     self.Vs[i].trainable_weights))
        policy_grad = grads[self.V_num * v_weights_len:self.V_num * v_weights_len + policy_weights_len]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        if self.is_attn:
            attn_start_dim = self.V_num * v_weights_len + policy_weights_len
            attn_weights_len = len(self.attn_net.trainable_weights)
            attn_grad = grads[attn_start_dim:attn_start_dim+attn_weights_len]
            self.attn_optimizer.apply_gradients(zip(attn_grad, self.attn_net.trainable_weights))

    @tf.function
    def apply_grads_all(self, grads):
        self.policy_optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        # TODO(guanyang): check if need to clip
        log_std = tf.clip_by_value(log_std, -5., -1.)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            actions = act_dist.sample()
            logps = act_dist.log_prob(actions)
            return actions, logps

    @tf.function
    def compute_logps(self, obs, actions):
        with self.tf.name_scope('compute_logps') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            actions = self.tf.clip_by_value(actions, -self.action_range+0.01, self.action_range-0.01)
            return act_dist.log_prob(actions)

    @tf.function
    def compute_entropy(self, obs):
        with self.tf.name_scope('compute_entropy') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            try:
                entropy = self.tf.reduce_mean(act_dist.entropy())
            except NotImplementedError:
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                entropy = -self.tf.reduce_mean(logps)
            finally:
                return entropy

    @tf.function
    def compute_kl(self, obs, other_out):  # KL(other||ego)
        with self.tf.name_scope('compute_entropy') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            other_act_dist = self._logits2dist(self.tf.stop_gradient(other_out))
            try:
                kl = self.tf.reduce_mean(other_act_dist.kl_divergence(act_dist))
            except NotImplementedError:
                other_actions = other_act_dist.sample()
                other_logps = other_act_dist.log_prob(other_actions)
                logps = self.compute_logps(obs, other_actions)
                kl = self.tf.reduce_mean(other_logps - logps)
            finally:
                return kl

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean

    @tf.function
    def compute_Vs(self, obs):
        with self.tf.name_scope('compute_Vs') as scope:
            return [self.tf.squeeze(V(obs), axis=1) for V in self.Vs]

    @tf.function
    def compute_attn(self, obs_others, mask):
        with self.tf.name_scope('compute_attn') as scope:
            return self.attn_net([obs_others, mask])  # return (logits, weights) tuple


Name2PolicyCls = dict(policy_with_qs=PolicyWithQs,
                      policy_with_vs=PolicyWithVs)

if __name__ == '__main__':
    pass
