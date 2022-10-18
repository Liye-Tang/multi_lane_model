#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from tensorflow.core.util import event_pb2

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

config = {
    "font.family": 'serif',  # 衬线字体
    "font.size": 9,
    "font.serif": ['times new roman'],  # 宋体
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

SimSun = FontProperties(
    fname='/home/yang/Software/anaconda3/envs/tf2/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimSun.ttf')

WINDOWSIZE = 5

CURRENT_DIR = os.path.dirname(__file__)


def help_func(env):
    alg_list = ['mpg', 'mpg2', 'ndpg', 'nadp', 'td3', 'sac']
    lbs = ['$\mathrm{MPG}$(混合加权)', '$\mathrm{MPG}$(混合状态)', '$\mathrm{DDPG}$',
           '$\mathrm{ADP}$', '$\mathrm{TD3}$', '$\mathrm{SAC}$']
    dir_str = os.path.dirname(CURRENT_DIR) + '/results/{env}'.format(env=env) + '/{}/data2plot'
    if env == 'aircraft':
        tag2plot = ['episode_return']
        goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
        lim = (-200, 20)
        fair_value = -20
    else:
        assert env == 'inverted_pendulum'
        tag2plot = []
        goal_perf_list = [-20, -10, -2, -1, -0.5, -0.1, -0.01]
        lim = []
        fair_value = -20
    return tag2plot, alg_list, lbs, goal_perf_list, dir_str, lim, fair_value


def plot_eval_results_of_all_alg_n_runs(env, dirs_dict_for_plot=None):
    tag2plot, alg_list, lbs, goal_perf_list, dir_str, lim, fair_value = help_func(env)
    df_list = []
    df_in_one_run_of_one_alg = {}
    for alg in alg_list:
        data2plot_dir = dir_str.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
            eval_file = os.path.join(eval_dir,
                                     [file_name for file_name in os.listdir(eval_dir) if file_name.startswith('events')][0])
            eval_summarys = tf.data.TFRecordDataset([eval_file])
            data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
            data_in_one_run_of_one_alg.update({'iteration': []})
            for eval_summary in eval_summarys:
                event = event_pb2.Event.FromString(eval_summary.numpy())
                for v in event.summary.value:
                    t = tf.make_ndarray(v.tensor)
                    for tag in tag2plot:
                        if tag == v.tag[11:]:
                            data_in_one_run_of_one_alg[tag].append(float(t))
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1/len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period]/10000. for i in range(len2)]
            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            if df_in_one_run_of_one_alg['episode_return'][0]>0:
                df_in_one_run_of_one_alg['episode_return'] = df_in_one_run_of_one_alg['episode_return'].map(lambda x: -x)
            for tag in tag2plot:
                df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
            df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    width_pt, height_pt = 200, 150
    axes_list = [0.18, 0.2, 0.78, 0.78]
    f = plt.figure(figsize=(pt2inch(width_pt), pt2inch(height_pt)), dpi=200)
    linewidth = 0.5
    save_dir = '/home/yang/Desktop/毕业论文/图/chapter3/实验/'
    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="episode_return", hue="algorithm",
                 data=total_dataframe, linewidth=linewidth, style="algorithm", ax=ax, legend=None)
    sns.lineplot(x=[0., 10.], y=[fair_value, fair_value], linewidth=linewidth, color='black', linestyle='--')
    print(ax.lines[0].get_data())
    ax.set_xlabel("迭代次数($\\times 10^4$)", fontproperties=SimSun)
    ax.set_ylabel('平均累计收益', fontproperties=SimSun)
    plt.xlim(0., 10.2)
    plt.ylim(*lim)
    plt.savefig(CURRENT_DIR + '/飞行器_平均累计收益.pdf')
    mycopyfile(CURRENT_DIR + '/飞行器_平均累计收益.pdf', save_dir)
    plt.clf()
    allresults = {}
    results2print = {}
    convergence = {}

    for alg, group in total_dataframe.groupby('algorithm'):
        allresults.update({alg: []})
        for ite, group1 in group.groupby('iteration'):
            mean = group1['episode_return'].mean()
            std = group1['episode_return'].std()
            allresults[alg].append((mean, std, ite))

    for alg, result in allresults.items():
        perf = [res for res in result if res[2] > 5.]
        mean, std, ite = sorted(perf, key=lambda x: x[0])[-1]
        results2print.update({alg: [mean, 2 * std]})

    interval = 15
    for alg in ['mpg', 'mpg2', 'ndpg', 'nadp', 'td3', 'sac']:
        asym_mean, _ = results2print[alg]
        l = []
        ite_list = []
        for mean, std, ite in allresults[alg]:
            if asym_mean - interval < mean < asym_mean + interval:
                l.append(True)
            else:
                l.append(False)
            ite_list.append(ite)
        for i in range(len(l)):
            if all(l[i:]):
                convergence.update({alg: ite_list[i]})
                break
    print(results2print)
    print(convergence)


def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def calculate_fair_case_path_tracking():
    delta_u, delta_y, delta_phi, r, delta, acc = 2, 1, 10*np.pi/180, 0.2, 0.1, 0.5
    r = -0.01*delta_u**2-0.04*delta_y**2-0.1*delta_phi**2-0.02*r**2-5*delta**2-0.05*acc**2
    print(100*r)


def calculate_fair_case_inverted():
    x, theta, x_dot, theta_dot = 1., 0.1, 0.1, 0.05
    r = -0.01*x**2-theta**2-0.1*x_dot**2-0.1*theta_dot**2
    print(100*r)


def calculate_fair_case_aircraft():
    alpha = 0.04
    r = -100*alpha**2
    print(100*r)


if __name__ == "__main__":
    env = 'aircraft'  # inverted_pendulum aircraft
    plot_eval_results_of_all_alg_n_runs(env)
    # calculate_fair_case_path_tracking()
    # calculate_fair_case_inverted()