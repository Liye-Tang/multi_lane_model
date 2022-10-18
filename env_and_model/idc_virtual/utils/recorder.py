#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/11
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: recorder.py
# =====================================
import shutil

import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

from env_and_model.idc_virtual.endtoend_env_utils import *

sns.set(style="darkgrid")

WINDOWSIZE = 15
STEER_RATIO = 16.6

config = {
    "font.family": 'serif',  # 衬线字体
    "font.size": 9,
    "font.serif": ['times new roman'],  # 宋体
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

SimSun = FontProperties(
    fname='/home/guanyang/anaconda3/envs/tf2/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimSun.ttf')


def pt2inch(pt):
    return pt/72


class Recorder(object):
    def __init__(self):
        self.val2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi', 'delta_y', 'delta_v', 'delta_phi',
                           'a_x', 'steer', 'cal_time', 'path_idx', 'path_values', 'ss_time', 'is_ss']
        self.data_across_all_episodes = []
        self.val_list_for_an_episode = []

    def reset(self,):
        if self.val_list_for_an_episode:
            self.data_across_all_episodes.append(self.val_list_for_an_episode)
        self.val_list_for_an_episode = []

    def record(self, obs, act, cal_time, path_idx, path_values, ss_time, is_ss):
        ego_info, tracking_info = obs[:Para.EGO_ENCODING_DIM], \
                                     obs[Para.EGO_ENCODING_DIM:Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM]
        v_x, v_y, r, x, y, phi = ego_info
        delta_y, delta_phi, delta_v = tracking_info[:3]
        steer, a_x = act[0]*Para.STEER_SCALE, act[1]*Para.ACC_SCALE + Para.ACC_SHIFT

        steer = steer * 180 / math.pi
        self.val_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, delta_y, delta_phi,
                                                      delta_v, a_x, steer, cal_time, path_idx, path_values, ss_time, is_ss]))

    def save(self, logdir):
        np.save(logdir + '/data_across_all_episodes.npy', np.array(self.data_across_all_episodes))

    def load(self, logdir):
        self.data_across_all_episodes = np.load(logdir + '/data_across_all_episodes.npy', allow_pickle=True)

    def plot_and_save_ith_episode_curves(self, i, curve_path):
        episode2plot = self.data_across_all_episodes[i]
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        plot_dict = data_dict.copy()
        # dict(v_x=[],  # 1
        #      phi=[],  # 2
        #      delta_y=[],  # 3
        #      delta_phi=[],
        #      r=[],  # 4
        #      delta_v=[],  # 5
        #      acc_cmd=[],
        #      acc_real=[],  # 6
        #      steer_cmd=[],
        #      steer_real=[],  # 7
        #      decision_time_ms=[],
        #      safety_shield_time_ms=[],  # 8
        #      path_idx=[],
        #      is_safe=[]  # 9
        #      )
        plot_dict['r'] = plot_dict['r'] * 180 / np.pi
        plot_dict['acc_cmd'] = plot_dict['a_x']
        plot_dict['steer_cmd'] = plot_dict['steer'] * STEER_RATIO * 180 / np.pi
        plot_dict['pure_dctime_ms'] = plot_dict['cal_time'] - plot_dict['ss_time']
        plot_dict['is_safe'] = 1 - plot_dict['is_ss']
        if os.path.exists(curve_path):
            shutil.rmtree(curve_path)
            os.makedirs(curve_path)
        else:
            os.makedirs(curve_path)
        time_line = np.array([0.1 * k for k in range(len(plot_dict['r']))])
        color1, color2, color3, color4 = 'k', 'r', 'b', 'g'
        style1, style2, style3, style4 = '-', '--', '-.', ':'
        linewidth = 0.5
        axes_list = [0.3, 0.3, 0.68, 0.5]
        for key, value in plot_dict.items():
            f_name = None
            f = plt.figure(figsize=(pt2inch(420 / 3), pt2inch(420 / 3 * (2 / 3))), dpi=200)
            if key == 'v_x':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('速度[$\mathrm{m/s}$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '速度'
            elif key == 'phi':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('航向角[$\degree$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '航向角'
            elif key == 'delta_y':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('位置误差[$\mathrm{m}$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '位置误差'
            elif key == 'delta_phi':
                pass
            elif key == 'r':
                ax = plt.axes(axes_list)
                ax.plot(time_line, plot_dict['delta_phi'], linewidth=linewidth, color=color1, linestyle=style1,
                        label='航向角误差')
                ax.plot(time_line, value, linewidth=linewidth, color=color2, linestyle=style2,
                        label='横摆角速度')
                ax.set_ylabel('角度[$\degree$]/角速度[$\mathrm{\degree/s}$]', fontproperties=SimSun)
                ax.yaxis.set_label_coords(-0.2, 0.3)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                ax.legend(ncol=2, frameon=False, prop={'family': 'SimSun', 'size': 9},
                          bbox_to_anchor=(1.05, 1.5), columnspacing=0.4, handletextpad=0.1,
                          handlelength=1)
                f_name = '角度和角速度'
            elif key == 'delta_v':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('速度误差[$\mathrm{m/s}$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '速度误差'
            elif key == 'acc_cmd':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('加速度[$\mathrm{m/s^2}$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '加速度'
            elif key == 'steer_cmd':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('方向盘转角[$\degree$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '方向盘转角'
            elif key == 'pure_dctime_ms':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('计算时间[$\mathrm{ms}$]', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                f_name = '计算时间'
            elif key == 'path_idx':
                pass
            elif key == 'is_safe':
                ax = plt.axes(axes_list)
                ax.plot(time_line, plot_dict['path_idx'], linewidth=linewidth, color=color1,
                        linestyle=style1, label='路径')
                ax.plot(time_line, value, linewidth=linewidth, color=color2,
                        linestyle=style2, label='护盾')
                ax.set_ylabel('路径/护盾', fontproperties=SimSun)
                ax.set_xlabel("时间[$\mathrm{s}$]", fontproperties=SimSun)
                ax.legend(ncol=2, frameon=False, prop={'family': 'SimSun', 'size': 9},
                          bbox_to_anchor=(1.05, 1.5))
                f_name = '路径和安全护盾'
            plt.savefig(curve_path + '/{}.pdf'.format(f_name))
            plt.close(f)
