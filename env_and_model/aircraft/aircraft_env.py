#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: path_tracking_env.py
# =====================================

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class AircraftDynamics(object):
    def __init__(self, if_model=False):
        self.if_model = if_model

    def f_xu(self, states, actions, tau):
        # alpha: the angle of attack, q: pitch rate, delta_e: elevator actuator voltage
        alpha, q, delta_e = states[:, 0], states[:, 1], states[:, 2]
        u = actions[:, 0]
        deriv = [-1.01887 * alpha + 0.90506 * q - 0.00215 * delta_e,
                 0.82225 * alpha - 1.07741 * q - 0.17555 * delta_e,
                 -delta_e + 5*u]
        deriv = tf.stack(deriv, axis=1)
        next_states = states + tau * deriv
        next_alpha, next_q, next_delta_e = next_states[:, 0], next_states[:, 1], next_states[:, 2]
        if not self.if_model:
            next_alpha += tfd.Normal(0.1*tau * tf.ones_like(next_alpha), 0.05*tau).sample()
        next_delta_e = tf.where(next_delta_e > 1.57, 1.57, next_delta_e)
        next_delta_e = tf.where(next_delta_e < -1.57, -1.57, next_delta_e)
        next_states = tf.stack([next_alpha, next_q, next_delta_e], axis=1)
        return next_states

    def compute_rewards(self, states, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            alpha, q, delta_e = states[:, 0], states[:, 1], states[:, 2]
            u = actions[:, 0]
            # TODO(guanyang): determine these values
            scale = dict(rew_alpha=-100., rew_q=0., rew_delta_e=0., rew_u=-0.5,
                         punish_alpha=100.)

            rew_alpha = tf.square(alpha)
            rew_q = tf.square(q)
            rew_delta_e = tf.square(delta_e)
            rew_u = tf.square(u)
            rewards = scale['rew_alpha'] * rew_alpha + scale['rew_q'] * rew_q + \
                      scale['rew_delta_e'] * rew_delta_e + scale['rew_u'] * rew_u

            # constraints
            # TODO(guanyang): determine these values
            punish_alpha = tf.where(alpha < 0.1, tf.square(alpha-0.1), tf.zeros_like(alpha))
            punish = scale['punish_alpha'] * punish_alpha

            reward_dict = dict(rewards=rewards,
                               punish=punish,
                               rew_alpha=rew_alpha,
                               rew_q=rew_q,
                               rew_delta_e=rew_delta_e,
                               rew_u=rew_u,
                               # punish_alpha=punish_alpha,
                               scaled_rew_alpha=scale['rew_alpha'] * rew_alpha,
                               scaled_rew_q=scale['rew_q'] * rew_q,
                               scaled_rew_delta_e=scale['rew_delta_e'] * rew_delta_e,
                               scaled_rew_u=scale['rew_u'] * rew_u,
                               # scaled_punish_alpha=scale['punish_alpha'] * punish_alpha,
                               )
        return reward_dict


class AircraftModel(object):  # all tensors
    def __init__(self, **kwargs):
        if_model = kwargs['if_model'] if 'if_model' in kwargs else True
        self.dynamics = AircraftDynamics(if_model=if_model)
        self.obses = None
        self.actions = None
        self.reward_dict = None
        # TODO(guanyang): determine the value
        self.tau = 0.05
        plt.ion()

    def reset(self, obses):
        self.obses = obses
        self.reward_dict = None

    def rollout(self, actions):
        self.actions = self.action_trans(actions)
        self.obses = self.dynamics.f_xu(self.obses, self.actions, self.tau)
        self.reward_dict = self.dynamics.compute_rewards(self.obses, self.actions)
        return self.obses, self.reward_dict

    def action_trans(self, actions):
        return 1. * actions

    def render(self,):
        reward_dict = {}
        for k, v in self.reward_dict.items():
            reward_dict[k] = v.numpy()
        render(self.obses.numpy(), self.actions.numpy(), reward_dict)
        plt.pause(0.001)
        plt.show()


class AircraftEnv(gym.Env):
    def __init__(self, num_agent, **kwargs):
        self.dynamics = AircraftDynamics(if_model=False)
        self.obses = None
        self.actions = None
        self.num_agent = num_agent
        self.dones = np.zeros((self.num_agent,), dtype=np.int)
        # TODO(guanyang): determine the value
        self.step_limit = 200
        self.agent_step = np.zeros((self.num_agent,), dtype=np.int)
        self.tau = 0.05
        self.num_agent = num_agent
        self.reward_dict = None
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf] * 3),
                                                high=np.array([np.inf] * 3),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1.]), dtype=np.float32)
        # TODO(guanyang): determine the obs_scale
        self.obs_scale = [3., 5., 2.]
        self.rew_scale, self.rew_shift = 1., 0.
        self.punish_scale = 1.
        plt.ion()

    def reset(self, **kwargs):
        if 'init_obs' in kwargs.keys():
            self.obses = kwargs.get('init_obs')
            self.agent_step = np.zeros((self.num_agent,), dtype=np.int)
        else:
            init_obses = np.random.uniform(-0.1, 0.1, (self.num_agent, 3)).astype(np.float32)
            init_step = np.zeros((self.num_agent,), dtype=np.int)
            if self.obses is None or 'total' in kwargs.keys():
                self.obses = init_obses
                self.agent_step = init_step
            else:
                self.obses = np.where(self.dones.reshape((-1, 1)) == 1, init_obses, self.obses)
                self.agent_step = np.where(self.dones == 1, init_step, self.agent_step)

        self.reward_dict = None
        return self.obses

    def step(self, actions):
        self.actions = self.action_trans(actions)
        self.obses = self.dynamics.f_xu(self.obses, self.actions, self.tau).numpy()
        self.reward_dict = self.dynamics.compute_rewards(self.obses, self.actions)
        for k, v in self.reward_dict.items():
            self.reward_dict[k] = v.numpy()
        self.dones = self.judge_done(self.obses)
        self.agent_step += 1
        info = {'reward_dict': self.reward_dict}
        return self.obses, self.reward_dict['rewards'], self.dones, info

    def judge_done(self, obses):
        alpha, q, delta_e = obses[:, 0], obses[:, 1], obses[:, 2]
        # TODO(guanyang): determine the condition
        done = (np.abs(alpha) > 30*3.14/180) | (np.abs(delta_e) > 1.57) | (np.abs(q) > .2) | (self.agent_step > self.step_limit)
        return done

    def action_trans(self, actions):
        actions = np.clip(actions, self.observation_space.low, self.observation_space.high)
        actions = 1. * actions
        return actions

    def render(self, mode='human'):
        render(self.obses, self.actions, self.reward_dict)
        plt.pause(0.001)
        plt.show()


def render(obses, actions, reward_dict):
    plt.clf()
    alpha, q, delta_e = obses[0, 0], obses[0, 1], obses[0, 2]
    point0x, point0y = 0, 0
    point1x, point1y = point0x + 10 * np.cos(alpha), \
                       point0y + 10 * np.sin(alpha)

    point2x, point2y = point0x - 3 * np.cos(alpha+delta_e), \
                       point0y - 3 * np.sin(alpha+delta_e)

    plt.title("Demo_model")
    ax = plt.axes(xlim=(-3, 10), ylim=(-10, 10))
    ax.add_patch(plt.Rectangle((-3, -10),
                               13, 20, edgecolor='black',
                               facecolor='none'))
    ax.axis('equal')
    plt.axis('off')
    ax.plot(point0x, point0y, 'b.')
    ax.plot([point0x, point1x], [point0y, point1y], color='b')
    ax.plot([point0x, point2x], [point0y, point2y], color='r')
    ax.plot(point1x, point1y, 'y.')
    ax.plot(point2x, point2y, 'k.')

    text_x, text_y_start = -10, 10
    ge = iter(range(0, 1000, 1))
    scale = 1
    ax.text(text_x, text_y_start - scale*next(ge), '--state--')
    ax.text(text_x, text_y_start - scale*next(ge), 'alpha: {:.2f}rad'.format(alpha))
    ax.text(text_x, text_y_start - scale*next(ge), 'q: {:.2f}rad/s'.format(q))
    ax.text(text_x, text_y_start - scale*next(ge), 'delta_e: {:.2f}rad'.format(delta_e))
    if actions is not None:
        ax.text(text_x, text_y_start - scale*next(ge), '--act--')
        ax.text(text_x, text_y_start - scale*next(ge), 'action: {:.2f}'.format(actions[0, 0]))
    if reward_dict is not None:
        ax.text(text_x, text_y_start - scale*next(ge), '--rew--')
        for key, val in reward_dict.items():
            ax.text(text_x, text_y_start - scale*next(ge), '{}: {:.4f}'.format(key, val[0]))
    return ax


def test_env():
    env = AircraftEnv(num_agent=1)
    obs = env.reset()
    action = np.array([[-0.5]], np.float32)
    for _ in range(1000):
        done = 0
        while not done:
            obs, reward, done, info = env.step(action)
            env.render()
        env.reset()


def test_model():
    env = AircraftEnv(num_agent=1)
    model = AircraftModel()
    obs = env.reset()
    model.reset(obs)
    actions = np.array([[-0,]], np.float32)
    for _ in range(200):
        model.rollout(actions)
        # print(model.obses[0][0])
        model.render()


if __name__ == '__main__':
    test_env()
    # test_model()
