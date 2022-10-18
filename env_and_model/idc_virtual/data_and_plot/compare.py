#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/5/15
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: compare.py
# =====================================

import numpy as np
import ray

from algorithm.utils.task_pool import TaskPool
from env_and_model.idc_virtual.e2e_planner.e2e_planner import E2ePlanner
from env_and_model.idc_virtual.em_planner.em_planner import EmPlanner
from env_and_model.idc_virtual.hierarchical_decision.hier_decision import HierarchicalDecision
from env_and_model.idc_virtual.utils.load_policy import get_args
from utils import *

CURRENT_DIR = os.path.dirname(__file__)


def run_episodes(parallel_worker=4):
    ray.init()
    exp_dir = WORKING_DIR + '/results/mpg2/experiment-'
    ite = 50000
    args = get_args(exp_dir)
    remote_planners = [ray.remote(HierarchicalDecision).options(num_cpus=1).remote(exp_dir, ite, args)
                       for _ in range(parallel_worker)]
    task_pool = TaskPool()
    for planner in remote_planners:
        task_pool.add(planner, planner.run_an_episode.remote())
    idc_episode_info_list = []
    while len(idc_episode_info_list) < 100:
        for worker, objID in task_pool.completed(blocking_wait=True):
            episode_info = ray.get(objID)
            idc_episode_info_list.append(episode_info)
            task_pool.add(worker, worker.run_an_episode.remote())

    exp_dir = WORKING_DIR + '/results/e2e/experiment-'
    ite = 50000
    args = get_args(exp_dir)
    remote_planners = [ray.remote(E2ePlanner).options(num_cpus=1).remote(exp_dir, ite, args)
                       for _ in range(parallel_worker)]
    task_pool = TaskPool()
    for planner in remote_planners:
        task_pool.add(planner, planner.run_an_episode.remote())
    e2e_episode_info_list = []
    while len(e2e_episode_info_list) < 100:
        for worker, objID in task_pool.completed(blocking_wait=True):
            episode_info = ray.get(objID)
            e2e_episode_info_list.append(episode_info)
            task_pool.add(worker, worker.run_an_episode.remote())

    remote_planners = [ray.remote(EmPlanner).options(num_cpus=1).remote() for _ in range(parallel_worker)]
    task_pool = TaskPool()
    for planner in remote_planners:
        task_pool.add(planner, planner.run_an_episode.remote())
    em_episode_info_list = []
    while len(em_episode_info_list) < 100:
        for worker, objID in task_pool.completed(blocking_wait=True):
            episode_info = ray.get(objID)
            em_episode_info_list.append(episode_info)
            task_pool.add(worker, worker.run_an_episode.remote())

    driving_performance_info = dict(idc=idc_episode_info_list,
                                    e2e=e2e_episode_info_list,
                                    em=em_episode_info_list)
    save_dir = os.path.abspath(__file__) + '/driving_performance_info.npy'
    np.save(save_dir, driving_performance_info)


def data_analysis():
    driving_performance_info = np.load(CURRENT_DIR + '/driving_performance_info.npy',
                                       allow_pickle=True)

    def info_extraction_from_an_episode(episode_info):
        # {'cal_time_ms': (time.time() - start_time) * 1000,
        #  'a_x': action[1] * Para.ACC_SCALE + Para.ACC_SHIFT,
        #  'pass_time_s': 0.1,
        #  'done': done,
        #  'done_type': self.env.done_type}
        # I time
        computing_time = np.array([info['cal_time_ms'] for info in episode_info])
        computing_time_mean, computing_time_std = computing_time.mean(), computing_time.std()
        # I_efficiency
        passing_time = 0.1*len(episode_info) if episode_info[-1]['done_type'] == 'good_done' else -1
        # I_comfort
        comfort = np.sqrt(np.square(np.array([info['ax'] for info in episode_info])).mean())
        # I_safety
        collision_with_other = 1 if episode_info[-1]['done_type'] == 'collision' else 0
        collision_with_road = 1 if episode_info[-1]['done_type'] == 'break_road_constrain' else 0
        collision = 1 if collision_with_other or collision_with_road else 0
        # I_compliance
        break_red_light = 1 if episode_info[-1]['done_type'] == 'break_red_light' else 0
        break_lane = 1 if episode_info[-1]['done_type'] == 'deviate_too_much' else 0
        break_speed_limit = 1 if episode_info[-1]['done_type'] == 'break_speed_limit' else 0
        traffic_rule_break = 1 if break_red_light + break_lane + break_speed_limit > 0 else 0
        # I_failure
        exceed_time_limit = 1 if episode_info[-1]['done_type'] == 'exceed_step_limit' else 0
        return dict(computing_time_mean=computing_time_mean,
                    computing_time_std=computing_time_std,
                    passing_time=passing_time,
                    comfort=comfort,
                    collision_with_other=collision_with_other,
                    collision_with_road=collision_with_road,
                    collision=collision,
                    break_red_light=break_red_light,
                    break_lane=break_lane,
                    break_speed_limit=break_speed_limit,
                    traffic_rule_break=traffic_rule_break,
                    exceed_time_limit=exceed_time_limit,)

    def get_statistical_result(info_list):
        if not info_list:
            return None
        I_time_mean = np.array([info['computing_time_mean'] for info in info_list]).mean()
        I_time_std = np.array([info['computing_time_std'] for info in info_list]).mean()
        I_traffic_mean = np.array([info['passing_time'] for info in info_list if info['passing_time'] > 0]).mean()
        I_traffic_std = np.array([info['passing_time'] for info in info_list if info['passing_time'] > 0]).std()
        I_comfort_mean = np.array([info['comfort'] for info in info_list]).mean()
        I_comfort_std = np.array([info['comfort'] for info in info_list]).std()
        I_safety = sum([info['collision'] for info in info_list])
        I_safety_other = sum([info['collision_with_other'] for info in info_list])
        I_safety_road = sum([info['collision_with_road'] for info in info_list])
        I_comp = sum([info['traffic_rule_break'] for info in info_list])
        I_comp_light = sum([info['break_red_light'] for info in info_list])
        I_comp_lane = sum([info['break_lane'] for info in info_list])
        I_comp_speed = sum([info['break_speed_limit'] for info in info_list])
        I_failure = sum([info['exceed_time_limit'] for info in info_list])
        return dict(I_time="I_time_mean: {}, I_time_std: {}".format(I_time_mean, I_time_std),
                    I_traffic="I_traffic_mean: {}, I_traffic_std: {}".format(I_traffic_mean, I_traffic_std),
                    I_comfort="I_comfort_mean: {}, I_comfort_std: {}".format(I_comfort_mean, I_comfort_std),
                    I_safety="I_safety: {}, I_safety_other: {}, I_safety_road: {}".format(I_safety, I_safety_other, I_safety_road),
                    I_comp="I_comp: {}, I_comp_light: {}, I_comp_lane: {}, I_comp_speed: {}".format(I_comp, I_comp_light, I_comp_lane, I_comp_speed),
                    I_failure="I_failure: {}".format(I_failure))

    for name in ['idc', 'e2e', 'em']:
        all_episode_info = [info_extraction_from_an_episode(episode_info) for episode_info in driving_performance_info[name]]
        print(name)
        print(get_statistical_result(all_episode_info))
