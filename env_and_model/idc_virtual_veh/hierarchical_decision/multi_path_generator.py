#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================
from env_and_model.idc_virtual.dynamics_and_models import ReferencePath


class MultiPathGenerator(object):
    def __init__(self):
        self.path_list = []

    def generate_path(self, task, green_or_red):
        ref = ReferencePath(task)
        task_path_num = len(ref.path_list['green'])
        self.path_list = []
        for path_index in range(task_path_num):
            ref = ReferencePath(task, green_or_red)
            ref.set_path(green_or_red, path_index)
            self.path_list.append(ref)
        return self.path_list

