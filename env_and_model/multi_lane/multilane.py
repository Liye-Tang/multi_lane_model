import random

import gym
import numpy as np
from gym.utils import seeding

import matplotlib.pyplot as plt


from env_and_model.multi_lane.car_tracking.ref_path import ReferencePath
from env_and_model.multi_lane.car_tracking.dynamics_and_models import VehicleDynamics, MultiLaneModel
from env_and_model.multi_lane.car_tracking.utils import *

import tensorflow as tf


class MultiLane(gym.Env):
    def __init__(self,
                 **kwargs
                 ):
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)    # TODO
        self.seed()

        self.obs = None
        self.action = None
        self.ego_state = None
        self.veh_list = []
        self.closest_point = None
        self.future_n_point = None
        self.ref_n_points = None
        self.area_index = None
        self.base_frequency = 10
        self.max_step = Para.MAX_STEP
        self.step_num = 0

        self.context = None
        self.lane_width = None
        self.lane_shape = None
        self.left_lane = None
        self.right_lane = None

        self.obs_scale = Para.OBS_SCALE
        self.rew_scale, self.rew_shift = 1., 0.
        self.punish_scale = 1.

        self.done_type = None
        self.done = False
        self.ref_v = 5

        self.env_model = MultiLaneModel()
        self.ego_dynamic = VehicleDynamics()
        plt.ion()

        self.ref_path = None
        if self.context is None:
            self.reset()
            self.render(mode='human')
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self.observation_space = convert_observation_to_space(self.obs)
        else:
            self.ref_path = ReferencePath(self.context)

    def step(self, action):
        self.step_num += 1
        info = {}
        self.action = action_denormalize(action)
        closest_point = np.array([i for i in self.closest_point])
        reward, reward_info = self.compute_reward(self.obs, self.action, closest_point)
        info.update(reward_info)
        self.ego_state, ego_param = self.get_next_ego_state(action)
        self.get_next_veh_list()
        self.update_obs()
        info.update({'closest_point': self.closest_point})
        self.done_type, self.done = self.judge_done(info)
        # TODO
        # print(reward)
        # if self.done == 2:
        #     reward = -100
        # return process_obs(self.obs), reward, self.done, info
        return self.obs, reward, self.done, info

    def reset(self, **kwargs):
        flag = 0
        while not flag:
            self.generate_context()
            self.generate_lane()
            self.ref_n_points, self.lane_shape = self.ref_path.get_random_n_points(Para.N)
            self.generate_ego_state()
            self.generate_veh_list()
            self.update_obs()
            flag = self.init_collision_check()
        self.step_num = 0
        # return process_obs(self.obs)
        return self.obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def compute_reward(self, obs, action, closest_point):
        obses, actions, closest_points = obs[np.newaxis, :], action[np.newaxis, :], closest_point[np.newaxis, :]
        reward, info_dict = self.env_model.compute_rewards(obses, actions)
        for k, v in info_dict.items():
            info_dict[k] = v.numpy()[0]
        reward_numpy = reward.numpy()[0]
        return reward_numpy, info_dict

    def get_next_ego_state(self, action):
        ego_states, actions = self.ego_state[np.newaxis, :], action[np.newaxis, :]
        next_ego_states, next_ego_params = self.ego_dynamic.prediction(ego_states, actions, Para.FREQUENCY)
        next_ego_state = next_ego_states.numpy()[0]
        next_ego_param = next_ego_params.numpy()[0]
        # next_ego_state, next_ego_param = next_ego_states.numpy()[0], next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_param

    def get_next_veh_list(self):

        for num in range(len(self.veh_list)):
            self.veh_list[num][0] += self.veh_list[num][3] / self.base_frequency * np.cos(self.veh_list[num][2] * np.pi / 180.)
            self.veh_list[num][1] += self.veh_list[num][3] / self.base_frequency * np.sin(self.veh_list[num][2] * np.pi / 180.)

    def judge_done(self, info):
        """
        :return:
         1: good done: enter area 2
         2: bad done: constraint violation
         3: max step
         4: not done
        """
        if self.area_index == 2:
            return 'good_done', 1
        if info['veh2road4training'] > 0:
            return 'collision with road', 2
        elif info['veh2veh4training'] > 0:
            return 'collision with veh', 2
        if self.step_num > self.max_step:
            return 'exceed the max step', 3
        else:
            return 'not_done', 0

    def update_obs(self):
        padding_veh = [-100, -100, 0, 0]
        while len(self.veh_list) < 6:
            self.veh_list.append(padding_veh)
        self.veh_list.sort(key=lambda x: x[0])
        veh_array = np.array(self.veh_list).reshape(-1)
        ego_pos = self.ego_state[-3:]
        self.closest_point, self.area_index, _, _ = self.ref_path.find_closest_point(ego_pos)  # TODO: find the true closest point
        k = np.tan(self.closest_point[2] * np.pi / 180)
        if abs(k) > 1000:
            min_dis = self.ego_state[3]
            is_left = self.ego_state[0] < self.closest_point[0]
        else:
            min_dis = cal_point_line_dis([self.ego_state[3], self.ego_state[4]], k,
                                         self.closest_point[0], self.closest_point[1])
            is_left = judge_point_line_pos((self.ego_state[0], self.ego_state[1]), k,
                                           self.closest_point[0], self.closest_point[1])
        devi_p = -min_dis if is_left else min_dis
        devi_phi = self.ego_state[5] - self.closest_point[2]

        devi_v = self.ego_state[0] - self.ref_v
        track_state = [devi_p, devi_phi, devi_v]
        obs = np.concatenate((self.ego_state, track_state, self.closest_point, veh_array, np.transpose(self.ref_n_points).reshape(-1),
                              [self.left_lane, self.right_lane, self.lane_width, self.ref_v]), axis=0)
        self.obs = convert_to_rela(obs)

    def _deviate_too_much(self):
        return True if cal_eu_dist(
            self.ego_state[-3],
            self.ego_state[-2],
            self.closest_point[0],
            self.closest_point[1]
        ) > 1.5 * self.lane_width or abs(self.ego_state[-1] - self.closest_point[2]) > Para.ANGLE_TOLERANCE else False

    def generate_context(self):
        goal_x = np.random.uniform(low=Para.GOAL_X_LOW, high=Para.GOAL_X_UP)
        goal_y = np.random.uniform(low=Para.GOAL_Y_LOW, high=Para.GOAL_Y_UP)
        if goal_x > 0:
            goal_phi = np.random.uniform(low=Para.GOAL_PHI_LOW, high=90)
        else:
            goal_phi = np.random.uniform(low=90, high=Para.GOAL_PHI_UP)
        self.context = goal_x, goal_y, goal_phi
        # self.context = 0, 80, 90

    def generate_ego_state(self):
        # init_ref_point_index = np.random.randint(Para.N - 1)
        init_ref_point_index = 0
        # ratio = np.random.random()
        # ref_x, ref_y = (1 - ratio) * self.ref_n_points[:2, init_ref_point_index] + \
        #                ratio * self.ref_n_points[:2, init_ref_point_index + 1]
        ref_x, ref_y = self.ref_n_points[:2, init_ref_point_index]
        ref_phi, ref_v = self.ref_n_points[2, init_ref_point_index], self.ref_v
        # add some noise
        ego_state = [0] * 6
        # ego_state[3] = ref_x + np.random.uniform(low=-(0.5+self.left_lane)*self.lane_width,
        #                                          high=(0.5+self.right_lane)*self.lane_width)
        # ego_state[4] = ref_y + np.random.uniform(low=-Para.Y_RANGE, high=Para.Y_RANGE)
        ego_state[3] = ref_x + np.random.uniform(low=self.lane_width*(0.35+self.left_lane),
                                                 high=self.lane_width*(0.35+self.right_lane))
        ego_state[4] = ref_y

        ego_state[5] = ref_phi + np.random.uniform(low=-10,
                                                   high=10)
        ego_state[0] = np.random.random() * ref_v
        ego_state[1] = 0
        ego_state[2] = np.random.random() * 2 - 1

        self.ego_state = np.array(ego_state, dtype=np.float32)

    def generate_veh_list(self):
        self.veh_list = []
        veh_mf, veh_mb, veh_lf, veh_lb, veh_rf, veh_rb = [None] * 6
        veh_mf = self.generate_vehicle4lane(0, 1) if np.random.random() > 0.3 else None
        veh_mb = self.generate_vehicle4lane(0, 0) if np.random.random() > 0.3 else None
        if self.left_lane:
            veh_lf = self.generate_vehicle4lane(-1, 1) if np.random.random() > 0.3 else None
            veh_lb = self.generate_vehicle4lane(-1, 0) if np.random.random() > 0.3 else None

        if self.right_lane:
            veh_rf = self.generate_vehicle4lane(1, 1) if np.random.random() > 0.3 else None
            veh_rb = self.generate_vehicle4lane(1, 0) if np.random.random() > 0.3 else None

        for veh in [veh_mf, veh_mb, veh_lf, veh_lb, veh_rf, veh_rb]:
            if veh is not None:
                self.veh_list.append(veh)

        # self.veh_list = []

    def generate_vehicle4lane(self, lane_encoding, front):
        veh_index = np.random.randint(Para.N-1)
        ref_x, ref_y, ref_phi = self.ref_n_points[:, veh_index]

        veh_state = [0] * 4
        veh_state[3] = np.random.random() * self.ref_v

        # get the base position
        if front:
            veh_state[0], veh_state[1], veh_state[2] = ref_x, ref_y, ref_phi
        else:
            if self.lane_shape:
                veh_state[0], veh_state[1], veh_state[2] = ref_x, -ref_y, 180 - ref_phi
            else:
                veh_state[0], veh_state[1], veh_state[2] = -ref_x, -ref_y, ref_phi

        # add some noise with some probability
        if np.random.random() < 0.2:
            veh_state[0] += np.random.uniform(low=-0.5*self.lane_width, high=0.5*self.lane_width)
            veh_state[2] += np.random.uniform(low=-30, high=30)
            # veh_state[2] = np.random.random() * ref_v

        # lane_encoding: left->-1 right->1 medium->0
        theta = (ref_phi - 90) / 180 * np.pi
        veh_state[0] += self.lane_width * np.cos(theta) * lane_encoding
        veh_state[2] += self.lane_width * np.sin(theta) * lane_encoding

        return veh_state

    def generate_lane(self):
        self.ref_path = ReferencePath(self.context)
        self.lane_width = np.random.uniform(3, 3.5)
        self.left_lane = np.random.uniform(0, 1) > 0.3
        self.right_lane = np.random.uniform(0, 1) > 0.3
        self.left_lane, self.right_lane = int(self.left_lane), int(self.right_lane)
        # self.lane_width = 3.75
        # self.left_lane, self.right_lane = 0, 0

    def init_collision_check(self):
        cum_cost = 0
        for i in range(1):
            action = np.array([0, 1/3])
            _, _, _, info = self.step(action)
            cum_cost += info['veh2veh4training'] + info['veh2road4training']
        return False if cum_cost > 0 else True

    def render(self, mode="human"):
        if mode == 'human':
            # basic render settings
            patches = []
            plt.clf()
            ax = plt.axes([0.05, 0.05, 0.9, 0.9])
            ax.axis('equal')

            # plot road typology
            ax.plot(self.ref_n_points[0], self.ref_n_points[1])
            plot_multi_lane(ax, self.ref_n_points[0], self.ref_n_points[1], self.ref_n_points[2],
                            self.left_lane, self.right_lane, self.lane_width)
            ax.scatter(self.closest_point[0], self.closest_point[1])

            # plot ego vehicle
            patches.append(
                draw_rotate_rec(self.ego_state[-3], self.ego_state[-2], self.ego_state[-1], Para.L, Para.W)
            )

            # plot sur vehicle
            veh_array = np.array(self.veh_list)
            veh_xs = veh_array[:, 0]
            veh_ys = veh_array[:, 1]
            veh_phis = veh_array[:, 2]
            patches.extend(draw_rotate_batch_rec(veh_xs, veh_ys, veh_phis, Para.L, Para.W))

            # plot closest point
            ax.scatter(self.closest_point[0], self.closest_point[1], color='r')

            # plot the whole fig
            ax.add_collection(PatchCollection(patches, match_original=True))
            plt.show()
            plt.pause(0.001)


def test():
    import tensorflow as tf
    env = MultiLane()
    env_model = MultiLaneModel()
    env.reset()
    i = 0
    while i < 1000:
        for j in range(200):
            i += 1
            # action = np.array([0, 0.6 + np.random.random() * 0.8], dtype=np.float32)  # np.random.rand(1)*0.1 - 0.05
            action = np.array([0, 1], dtype=np.float32)  # np.random.rand(1)*0.1 - 0.05
            obs, reward, done, info = env.step(action)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # print(reward)
            # obses = tf.convert_to_tensor(np.tile(obs, (1, 1)), dtype=tf.float32)
            # ref_points = tf.convert_to_tensor(np.tile(info['future_n_point'], (1, 1, 1)), dtype=tf.float32)
            # actions = tf.convert_to_tensor(np.tile(actions, (1, 1)), dtype=tf.float32)
            # env_model.reset(obses, ref_points)
            env.render()
            # if j > 88:
            #     for i in range(25):
            #         obses, rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
            #         veh2bike4real, veh2person4real, veh2speed4real = env_model.rollout_out(
            #             actions + tf.experimental.numpy.random.rand(2) * 0.05, i)
            #         env_model.render()
            if done:
                print(env.done_type)
                break
        env.reset()
        # env.render(weights=np.zeros(env.other_number,))


def test_model():
    import tensorflow as tf
    env = MultiLane()
    env_model = MultiLaneModel()
    obs = env.reset()
    obses = tf.expand_dims(obs, axis=0)
    actions = tf.zeros((1, 2))
    env_model.reset(obses)
    for i in range(100):
        env_model.rollout(actions)


if __name__ == '__main__':
    test()
