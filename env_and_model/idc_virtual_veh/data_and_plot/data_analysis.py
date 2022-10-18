#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/5/14
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: data_analysis.py
# =====================================
import numpy as np

from env_and_model.idc_virtual_veh.endtoend_env_utils import *
from utils import *


def pt2inch(pt):
    return pt/72


def exp2_1():
    mpc_info = np.load(os.path.dirname(__file__) + '/info_from_same_start_mpc_2022-05-17-19-24-04.npy', allow_pickle=True)
    rl_info = np.load(os.path.dirname(__file__) + '/info_from_same_start_rl_2022-05-17-19-24-04.npy', allow_pickle=True)
    # for i, mpc_epi_info in enumerate(mpc_info):
    #     for j, info in enumerate(mpc_epi_info):
    #         print('mpc epi {}, step {}'.format(i, j), info)
    # for i, rl_epi_info in enumerate(rl_info):
    #     for j, info in enumerate(rl_epi_info):
    #         print('rl epi {}, step {}'.format(i, j), info)
    mpc_rewards = np.array([sum([info['rewards4value'] for info in mpc_epi_info]) for mpc_epi_info in mpc_info])
    mpc_punish = np.array([sum([info['real_punish_term'] for info in mpc_epi_info]) for mpc_epi_info in mpc_info])

    rl_rewards = np.array([sum([info['rewards4value'] for info in rl_epi_info]) for rl_epi_info in rl_info])
    rl_punish = np.array([sum([info['real_punish_term'] for info in rl_epi_info]) for rl_epi_info in rl_info])
    print('mpc_rewards', mpc_rewards)
    print('mpc_punish', mpc_punish)
    print('rl_rewards', rl_rewards)
    print('rl_punish', rl_punish)
    print('mpc_reward_mean: ', mpc_rewards.mean(), 'mpc_reward_std: ', mpc_rewards.std())
    print('mpc_punish_mean: ', mpc_punish.mean(), 'mpc_punish_std: ', mpc_punish.std())

    print('rl_reward_mean: ', rl_rewards.mean(), 'rl_reward_std: ', rl_rewards.std())
    print('rl_punish_mean: ', rl_punish.mean(), 'rl_punish_std: ', rl_punish.std())

    mpc_rewards_new = []
    for mpc_rew, mpc_pun in zip(mpc_rewards, mpc_punish):
        if mpc_rew > 200 or mpc_pun > 0:
            pass
        else:
            mpc_rewards_new.append(mpc_rew*0.1)

    rl_rewards_new = []
    for rl_rew, rl_pun in zip(rl_rewards, rl_punish):
        if rl_rew > 200 or rl_pun > 0:
            pass
        else:
            rl_rewards_new.append(rl_rew*0.1)
    mpc_rewards_new, rl_rewards_new = np.array(mpc_rewards_new), np.array(rl_rewards_new)
    print('mpc_rewards_new: ', mpc_rewards_new)
    print('rl_rewards_new: ', rl_rewards_new)
    print('mpc_reward_new_mean: ', mpc_rewards_new.mean(), 'mpc_reward_new_std: ', mpc_rewards_new.std())
    print('rl_reward_new_mean: ', rl_rewards_new.mean(), 'rl_reward_new_std: ', rl_rewards_new.std())


def exp2_2():
    all_runs = np.load(os.path.dirname(__file__) + '/all_runs_2022-05-18-00-06-56.npy', allow_pickle=True)
    # mpc_path_values = mpc_path_values,
    # mpc_path_index = mpc_path_index,
    # mpc_action = mpc_action,
    # mpc_time = self.mpc_cal_timer.mean())
    # adp_path_values=adp_path_values,
    # adp_path_index=adp_path_index,
    # adp_action=adp_action,
    # adp_time=self.adp_cal_timer.mean()))
    # done = done,
    # done_type = self.env.done_type
    all_steps = np.array(sum([[1 if info['mpc_path_index'] == info['adp_path_index'] else 0 for info in run]
                     for run in all_runs], []))
    mpc_time = np.array(sum([[info['mpc_time'] for info in run if info['mpc_time'] < 2500] for run in all_runs], []))
    adp_time = np.array(sum([[info['adp_time'] for info in run] for run in all_runs], []))
    print(mpc_time.max())
    print('all_step: ', len(all_steps), all_steps.sum(), all_steps.mean())
    print(list(mpc_time))
    print(list(adp_time))
    print('mpc_time_mean: ', mpc_time.mean(), 'mpc_time_std: ', mpc_time.std())
    print('adp_time_mean: ', adp_time.mean(), 'adp_time_std: ', adp_time.std())
    print(mpc_time.mean()/adp_time.mean())


if __name__ == '__main__':
    # exp2_1()
    exp2_2()

