#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ray

from algorithm.tester import Tester
from algorithm.trainer import Trainer
from env_and_model import Name2EnvAndModelCls
from algorithm.buffer import Name2BufferCls
from algorithm.evaluator import Name2EvaluatorCls
from algorithm.policy import Name2PolicyCls
from algorithm.learners import Name2LearnerCls
from algorithm.optimizer import Name2OptimizerCls
from algorithm.worker import Name2WorkerCls


def built_AMPC_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='training')  # training testing debug
    known_args, _ = parser.parse_known_args(['mode'])
    mode = known_args.mode

    if mode == 'testing':
        test_dir = os.path.dirname(os.path.dirname(__file__)) + '/results/ampc/ampc-experiment-2022-08-30-10-51-58'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = test_dir + '/logs/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[295000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=20,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=200,
                           eval_render=True))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--learner_type', type=str, default='ampc')
    parser.add_argument('--policy_type', type=str, default='policy_with_qs')
    parser.add_argument('--worker_type', type=str, default='offpolicy')
    parser.add_argument('--evaluator_type', type=str, default='evaluator')
    parser.add_argument('--buffer_type', type=str, default='rb')
    parser.add_argument('--optimizer_type', type=str, default='offpolicy')
    parser.add_argument('--off_policy', type=str, default=True)
    parser.add_argument('--memo', type=str, default='')
    parser.add_argument('--max_iter', type=int, default=1000000)
    known_args, _ = parser.parse_known_args(['max_iter'])
    max_iter = known_args.max_iter
    # env
    parser.add_argument('--env_id', default='multi_lane',
                        help="['MultiLane']")
    # TODO(guanyang): tbd
    parser.add_argument('--num_agent', type=int, default=8)
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)

    # specially for idc env
    parser.add_argument('--is_attn', type=bool, default=False)
    parser.add_argument('--state_dim', default=None)
    parser.add_argument('--other_start_dim', type=int, default=None)
    parser.add_argument('--attn_lr_schedule', type=list, default=None)
    parser.add_argument('--attn_in_total_dim', type=int, default=None)
    parser.add_argument('--attn_in_per_dim', type=int, default=None)
    parser.add_argument('--attn_out_dim', type=int, default=None)

    # learner
    parser.add_argument('--alg_name', default='AMPC')
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--num_rollout_for_policy_update', type=int, default=25)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--init_punish_factor', type=float, default=0.)
    parser.add_argument('--pf_enlarge_interval', type=int, default=20000)
    parser.add_argument('--pf_amplifier', type=float, default=0.)

    # worker
    parser.add_argument('--sample_batch_size', type=int, default=64 if mode == 'debug' else 512)
    parser.add_argument('--worker_log_interval', type=int, default=100)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=64 if mode == 'debug' else 3000)
    parser.add_argument('--replay_batch_size', type=int, default=64 if mode == 'debug' else 512)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=10)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=100)
    parser.add_argument('--eval_render', type=bool, default=False)
    known_args, _ = parser.parse_known_args(['num_eval_episode'])
    num_eval_episode = known_args.num_eval_episode
    parser.add_argument('--num_eval_agent', type=int, default=num_eval_episode)

    # policy and model
    parser.add_argument('--value_model_cls', type=str, default='mlp')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='gelu')
    parser.add_argument('--value_out_activation', type=str, default='linear')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, max_iter, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='mlp')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='gelu')
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, max_iter, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--Q_num', type=int, default=0)
    parser.add_argument('--target', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=None)

    # preprocessor
    parser.add_argument('--obs_scale', type=list, default=None)
    parser.add_argument('--rew_scale', type=float, default=None)
    parser.add_argument('--rew_shift', type=float, default=None)
    parser.add_argument('--punish_scale', type=float, default=None)

    # optimizer (PABAL)
    parser.add_argument('--num_workers', type=int, default=1 if mode == 'debug' else 1)
    parser.add_argument('--num_learners', type=int, default=2 if mode == 'debug' else 2)
    parser.add_argument('--num_buffers', type=int, default=1 if mode == 'debug' else 1)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=25)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = working_dir + '/results/ampc/ampc-experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)

    return parser.parse_args()


def built_parser():
    args = built_AMPC_parser()
    if args.mode == 'testing':
        return args
    env_id = args.env_id
    env_cls, _ = Name2EnvAndModelCls[env_id]
    env = env_cls(num_agent=1)  # for idc env, the "num_agent" key has no influence
    # fill common arguments
    args.obs_dim, args.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    args.obs_scale, args.rew_scale = env.obs_scale, env.rew_scale
    args.rew_shift, args.punish_scale = env.rew_shift, env.punish_scale
    env.close()
    return args


def main():
    args = built_parser()
    logger.info('Parameters: {}'.format(str(args)))
    # working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd_list = ['#!/bin/bash\n', 'eval "$(conda shell.bash hook)"\n', 'conda activate tf2\n', 'cd ' + args.result_dir + '\n',
                'tensorboard --logdir ./logs\n', 'chmod +x display.sh']
    if args.mode == 'training' or args.mode == 'debug':
        ray.init(object_store_memory=5120*1024*1024)
        os.makedirs(args.result_dir)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        with open(args.result_dir + '/display.sh', 'w', encoding='utf-8') as f:
            [f.write(cmd) for cmd in cmd_list]
        trainer = Trainer(policy_cls=Name2PolicyCls[args.policy_type],
                          worker_cls=Name2WorkerCls[args.worker_type],
                          learner_cls=Name2LearnerCls[args.learner_type],
                          buffer_cls=Name2BufferCls[args.buffer_type],
                          optimizer_cls=Name2OptimizerCls[args.optimizer_type],
                          evaluator_cls=Name2EvaluatorCls[args.evaluator_type],
                          args=args)
        if args.model_load_dir is not None:
            logger.info('loading model')
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        trainer.train()

    elif args.mode == 'testing':
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        with open(args.test_log_dir + '/display.sh', 'w', encoding='utf-8') as f:
            [f.write(cmd) for cmd in cmd_list]
        tester = Tester(policy_cls=Name2PolicyCls[args.policy_type],
                        evaluator_cls=Name2EvaluatorCls[args.evaluator_type],
                        args=args)
        tester.test()


if __name__ == '__main__':
    main()
