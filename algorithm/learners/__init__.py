#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/1/4
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: __init__.py.py
# =====================================

from algorithm.learners.ampc import AMPCLearner, AMPCLearnerWithAttention


Name2LearnerCls = dict(ampc=AMPCLearner,
                       ampc_with_attn=AMPCLearnerWithAttention)