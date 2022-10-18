#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
import os

import ray

from algorithm.buffer import ReplayBufferWithAttention
from algorithm.evaluator import EvaluatorWithAttention
from algorithm.ampc import AMPCLearnerWithAttention
from algorithm.optimizer import OffPolicyAsyncOptimizer, SingleProcessOffPolicyOptimizer
from algorithm.policy import AttentionPolicy4Toyota
from algorithm.tester import Tester
from algorithm.trainer import Trainer
from algorithm.worker import OffPolicyWorkerWithAttention
from algorithm.utils.misc import args2envkwargs

from env_build.endtoend import CrossroadEnd2endMix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NAME2WORKERCLS = dict([('OffPolicyWorkerWithAttention', OffPolicyWorkerWithAttention)])
NAME2LEARNERCLS = dict([('AMPCWithAttention', AMPCLearnerWithAttention)])
NAME2BUFFERCLS = dict([('normalWithAttention', ReplayBufferWithAttention), ('None', None)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer),
                          ('SingleProcessOffPolicy', SingleProcessOffPolicyOptimizer)])
NAME2POLICIES = dict([('AttentionPolicy4Toyota', AttentionPolicy4Toyota)])
NAME2EVALUATORS = dict([('None', None), ('EvaluatorWithAttention', EvaluatorWithAttention)])


def built_AMPC_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training')  # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/CrossroadEnd2endMix-v0/experiment-2021-10-14-08-15-20'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[190000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=10,
                           eval_log_interval=1,
                           fixed_steps=150,
                           eval_render=True))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='AttentionPolicy4Toyota')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorkerWithAttention')
    parser.add_argument('--evaluator_type', type=str, default='EvaluatorWithAttention')
    parser.add_argument('--buffer_type', type=str, default='normalWithAttention')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='CrossroadEnd2endMix-v0')
    parser.add_argument('--max_step', type=int, default=200)
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument('--state_dim', default=None)

    parser.add_argument('--other_start_dim', type=int, default=None)
    parser.add_argument('--per_other_dim', type=int, default=None)
    parser.add_argument('--other_number', type=int, default=None)

    # learner
    parser.add_argument('--alg_name', default='AMPCWithAttention')
    parser.add_argument('--num_rollout_for_policy_update', type=list, default=25)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--init_punish_factor', type=float, default=20.)
    parser.add_argument('--pf_enlarge_interval', type=int, default=20000)
    parser.add_argument('--pf_amplifier', type=float, default=1.)

    # worker
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=50000)
    parser.add_argument('--replay_starts', type=int, default=1500)  # use a small value for debug
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=2)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=50)
    parser.add_argument('--eval_render', type=bool, default=False)

    # policy and model
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_lr_schedule', type=list, default=[1e-4, 300000, 2e-6])
    parser.add_argument('--value_lr_schedule', type=list, default=[3e-4, 300000, 1e-6])
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_hidden_units', type=int, default=256)
    parser.add_argument('--hidden_activation', type=str, default='gelu')
    parser.add_argument('--deterministic_policy', default=True)
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--action_range', type=float, default=None)

    # model for attn_net
    parser.add_argument('--attn_model_cls', type=str, default='Attention')
    parser.add_argument('--attn_in_per_dim', type=int, default=None)
    parser.add_argument('--attn_in_total_dim', type=int, default=None)
    parser.add_argument('--attn_out_dim', type=int, default=64)
    parser.add_argument('--attn_lr_schedule', type=list, default=[1e-4, 300000, 2e-6])

    # preprocessor
    parser.add_argument('--obs_scale', type=list, default=None)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    parser.add_argument('--reward_shift', type=float, default=0.)

    # optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=300000)
    parser.add_argument('--num_workers', type=int, default=10)  # use a small value for debug
    parser.add_argument('--num_learners', type=int, default=18)
    parser.add_argument('--num_buffers', type=int, default=10)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=500)

    # IO
    args = parser.parse_args()
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/{env}/experiment-{time}'.format(env=args.env_id, time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)

    return parser.parse_args()


def built_attention_parser(alg_name):
    if alg_name == 'AMPC':
        args = built_AMPC_parser()
        env = CrossroadEnd2endMix()
        obs_space, act_space = env.observation_space, env.action_space
        args.other_start_dim = env.other_start_dim
        args.attn_in_per_dim = env.per_other_info_dim
        args.attn_in_total_dim = env.per_other_info_dim * env.other_number
        args.obs_dim, args.act_dim = obs_space.shape[0], act_space.shape[0]
        args.state_dim = env.other_start_dim + args.attn_out_dim
        args.obs_scale = [0.2, 0., 2., 1 / 30., 1 / 30, 1 / 180.] + \
                         [1., 1 / 15., 0.2] + \
                         [1., 1.] + \
                         [1., 1., 1.] + \
                         [1., 1., 1.] + \
                         [1., 1., 1., 1.] + \
                         [1 / 30., 1 / 30., 0.2, 1 / 180., 0.2, 0.5, 1., 1., 1., 0.] * args.other_number
        env.close()
        return args


def main(alg_name):
    args = built_attention_parser(alg_name)
    logger.info('begin training agents with parameter {}'.format(str(args)))
    if args.mode == 'training':
        ray.init(object_store_memory=5120 * 1024 * 1024)
        os.makedirs(args.result_dir)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(policy_cls=NAME2POLICIES[args.policy_type],
                          worker_cls=NAME2WORKERCLS[args.worker_type],
                          learner_cls=NAME2LEARNERCLS[args.alg_name],
                          buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                          optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
                          evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                          args=args)
        if args.model_load_dir is not None:
            logger.info('loading model')
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        trainer.train()

    elif args.mode == 'testing':
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        tester = Tester(policy_cls=NAME2POLICIES[args.policy_type],
                        evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                        args=args)
        tester.test()


if __name__ == '__main__':
    main('AMPC')
