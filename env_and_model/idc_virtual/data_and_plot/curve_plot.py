#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from tensorflow.core.util import event_pb2

from utils import *

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

WINDOWSIZE = 15

CURRENT_DIR = os.path.dirname(__file__)


def pt2inch(pt):
    return pt/72

def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def plot_opt_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['optimizer/learner_stats/scalar/obj_loss',
                'optimizer/learner_stats/scalar/q_loss',
                'optimizer/learner_stats/scalar/policy_loss',
                'optimizer/learner_stats/scalar/pf',
                'optimizer/learner_stats/scalar/con_loss',
                'optimizer/learner_stats/scalar/real_punish_term_sum_model',
                # 'optimizer/learner_stats/scalar/punish_veh2road4real',
                # 'optimizer/learner_stats/scalar/punish_veh2veh4real',
                ]
    alg_list = ['ampc']
    data2plot_dir = WORKING_DIR + '/results/{}/data2plot'
    df_list = []
    for alg in alg_list:
        data2plot_dir = data2plot_dir.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            opt_dir = data2plot_dir + '/' + dir + '/logs/optimizer'
            opt_file = os.path.join(opt_dir,
                                     [file_name for file_name in os.listdir(opt_dir) if
                                      file_name.startswith('events')][0])
            opt_summarys = tf.data.TFRecordDataset([opt_file])
            data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
            data_in_one_run_of_one_alg.update({'iteration': []})
            for opt_summary in opt_summarys:
                event = event_pb2.Event.FromString(opt_summary.numpy())
                for v in event.summary.value:
                    t = tf.make_ndarray(v.tensor)
                    for tag in tag2plot:
                        if tag == v.tag:
                            data_in_one_run_of_one_alg[tag].append(float(t))
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1 / len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                       i in range(len2)]
            data_in_one_run_of_one_alg = {key: val for key, val in data_in_one_run_of_one_alg.items()}
            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            print({key: len(val) for key, val in data_in_one_run_of_one_alg.items() if hasattr(val, '__len__')})
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            for tag in tag2plot:
                df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
            df_in_one_run_of_one_alg['optimizer/learner_stats/scalar/obj_loss_smo'] = \
                df_in_one_run_of_one_alg['optimizer/learner_stats/scalar/obj_loss_smo'].map(lambda x: -x)
            df_list.append(df_in_one_run_of_one_alg)
    total_df = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]

    width_pt, height_pt = 200, 150
    axes_list = [0.2, 0.2, 0.8, 0.8]
    f = plt.figure(figsize=(pt2inch(width_pt), pt2inch(height_pt)), dpi=200)
    color1, color2, color3, color4 = 'k', 'r', 'b', 'g'
    style1, style2, style3, style4 = '-', '--', '-.', ':'
    save_dir = '/home/yang/Desktop/毕业论文/图/chapter4/实验/'

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="optimizer/learner_stats/scalar/q_loss_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("Q网络性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/Q网络性能曲线.pdf')
    mycopyfile(CURRENT_DIR + '/Q网络性能曲线.pdf', save_dir)
    plt.clf()

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="optimizer/learner_stats/scalar/con_loss_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("约束性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/约束性能曲线.pdf')
    mycopyfile(CURRENT_DIR + '/约束性能曲线.pdf', save_dir)
    plt.clf()

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="optimizer/learner_stats/scalar/obj_loss_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("目标性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/目标性能曲线.pdf')
    mycopyfile(CURRENT_DIR + '/目标性能曲线.pdf', save_dir)
    plt.clf()

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="optimizer/learner_stats/scalar/real_punish_term_sum_model_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("约束性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/约束性能曲线-真实.pdf')
    mycopyfile(CURRENT_DIR + '/约束性能曲线-真实.pdf', save_dir)
    plt.clf()


def plot_eva_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['evaluation/episode_return',
                'evaluation/punish_sum',
                'evaluation/real_punish_term_sum',
                ]
    alg_list = ['ampc']
    data2plot_dir = WORKING_DIR + '/results/{}/data2plot'
    df_list = []
    for alg in alg_list:
        data2plot_dir = data2plot_dir.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            eva_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
            eva_file = os.path.join(eva_dir,
                                     [file_name for file_name in os.listdir(eva_dir) if
                                      file_name.startswith('events')][0])
            eva_summarys = tf.data.TFRecordDataset([eva_file])
            data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
            data_in_one_run_of_one_alg.update({'iteration': []})
            for opt_summary in eva_summarys:
                event = event_pb2.Event.FromString(opt_summary.numpy())
                for v in event.summary.value:
                    t = tf.make_ndarray(v.tensor)
                    for tag in tag2plot:
                        if tag == v.tag:
                            data_in_one_run_of_one_alg[tag].append(float(t))
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1 / len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                       i in range(len2)]
            data_in_one_run_of_one_alg = {key: val for key, val in data_in_one_run_of_one_alg.items()}
            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            for tag in tag2plot:
                df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
            df_in_one_run_of_one_alg['evaluation/episode_return_smo'] = \
                df_in_one_run_of_one_alg['evaluation/episode_return_smo'].map(lambda x: -x)
            df_list.append(df_in_one_run_of_one_alg)
    total_df = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]

    width_pt, height_pt = 200, 150
    axes_list = [0.2, 0.2, 0.8, 0.8]
    f = plt.figure(figsize=(pt2inch(width_pt), pt2inch(height_pt)), dpi=200)
    color1, color2, color3, color4 = 'k', 'r', 'b', 'g'
    style1, style2, style3, style4 = '-', '--', '-.', ':'
    save_dir = '/home/yang/Desktop/毕业论文/图/chapter4/实验/'

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="evaluation/punish_sum_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("约束性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/eva约束性能曲线.pdf')
    mycopyfile(CURRENT_DIR + '/eva约束性能曲线.pdf', save_dir)
    plt.clf()

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="evaluation/episode_return_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("目标性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/eva目标性能曲线.pdf')
    mycopyfile(CURRENT_DIR + '/eva目标性能曲线.pdf', save_dir)
    plt.clf()

    ax = plt.axes(axes_list)
    sns.lineplot(x="iteration", y="evaluation/real_punish_term_sum_smo", data=total_df, ax=ax,
                 color=color1, linestyle=style1)
    ax.set_xlabel('迭代次数 [$\\times 10^4$]', fontproperties=SimSun)
    ax.set_ylabel("约束性能", fontproperties=SimSun)
    plt.savefig(CURRENT_DIR + '/eva约束性能曲线-真实.pdf')
    mycopyfile(CURRENT_DIR + '/eva约束性能曲线-真实.pdf', save_dir)
    plt.clf()


if __name__ == "__main__":
    plot_opt_results_of_all_alg_n_runs()
    # plot_eva_results_of_all_alg_n_runs()
