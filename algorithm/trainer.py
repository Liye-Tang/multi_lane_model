#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: trainer.py
# =====================================

import logging

import ray

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer(object):
    def __init__(self, policy_cls, worker_cls, learner_cls, buffer_cls, optimizer_cls, evaluator_cls, args):
        self.args = args
        if self.args.optimizer_type.startswith('singleprocess'):
            self.evaluator = evaluator_cls(policy_cls, self.args) \
                if evaluator_cls is not None else None
            if self.args.off_policy:
                self.local_worker = worker_cls(self.args, 0)
                self.buffer = buffer_cls(self.args, 0)
                self.learner = learner_cls(args)
                self.optimizer = optimizer_cls(self.local_worker, self.learner, self.buffer, self.evaluator, self.args)
            else:
                self.local_worker = worker_cls(learner_cls, self.args, 0)
                self.optimizer = optimizer_cls(self.local_worker, self.evaluator, self.args)

        else:
            self.evaluator = ray.remote(evaluator_cls).options(num_cpus=1).remote(policy_cls, self.args)
            if self.args.off_policy:
                self.local_worker = worker_cls(self.args, 0)
                self.remote_workers = [
                    ray.remote(worker_cls).options(num_cpus=1).remote(self.args, i + 1)
                    for i in range(self.args.num_workers)]
                self.workers = dict(local_worker=self.local_worker,
                                    remote_workers=self.remote_workers)
                self.buffers = [ray.remote(buffer_cls).options(num_cpus=1).remote(self.args, i+1)
                                for i in range(self.args.num_buffers)]
                self.learners = [ray.remote(learner_cls).options(num_cpus=1).remote(args)
                                 for _ in range(self.args.num_learners)]
                self.optimizer = optimizer_cls(self.workers, self.learners, self.buffers, self.evaluator, self.args)
            else:
                self.local_worker = worker_cls(learner_cls, self.args, 0)
                self.remote_workers = [
                    ray.remote(worker_cls).options(num_cpus=1).remote(learner_cls, self.args, i+1)
                    for i in range(self.args.num_workers)]
                self.workers = dict(local_worker=self.local_worker,
                                    remote_workers=self.remote_workers)
                self.optimizer = optimizer_cls(self.workers, self.evaluator, self.args)

    def load_weights(self, load_dir, iteration):
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.local_worker.load_weights(load_dir, iteration)
        else:
            self.local_worker.load_weights(load_dir, iteration)
            self.sync_remote_workers()

    def sync_remote_workers(self):
        weights = ray.put(self.local_worker.get_weights())
        for e in self.workers['remote_workers']:
            e.set_weights.remote(weights)

    def train(self):
        logger.info('training beginning')
        while self.optimizer.iteration < self.args.max_iter:
            self.optimizer.step()
        self.optimizer.stop()
        logger.info('training complete\n')
