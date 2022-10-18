#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/5/15
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: utils.py
# =====================================
import os
import shutil

WORKING_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.dirname(__file__) + '/results'
THESIS_FIG_DIR = '/home/yang/Desktop/毕业论文/图'


def pt2inch(pt):
    return pt/72


def mycopyfile(srcfile, dstpath, name=None):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        fname = name if name is not None else fname
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))
