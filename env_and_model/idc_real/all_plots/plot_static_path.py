#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/4/27
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: plot_static_path.py
# =====================================

from env_and_model.idc_real.dynamics_and_models import ReferencePath
from env_and_model.idc_real.endtoend_env_utils import *
from env_and_model.idc_real.all_plots.data_replay.render_utils import *


def plot_map():
    for light_phase, light in zip([0, 2, 4], ['绿灯', '黄灯', '红灯']):
        f = plt.figure(figsize=(pt2inch(420 / 3), pt2inch(420 / 3)), dpi=300)
        render(light_phase=light_phase, all_other=None, interested_other=None,
               attn_weights=None, obs=None, ref_path=None, future_n_point=None, action=None,
               done_type=None, reward_info=None, hist_posi=None, path_values=None, is_debug=False)
        plt.savefig('/home/yang/Desktop/毕业论文/图/chapter5/地图{}.png'.format(light))
        plt.show()
        plt.close(f)


def plot_static_path():
    f = plt.figure(figsize=(pt2inch(420 / 2), pt2inch(420 / 2)), dpi=300)
    ax = render(light_phase=0, all_other=None, interested_other=None,
                attn_weights=None, obs=None, ref_path=None, future_n_point=None, action=None,
                done_type=None, reward_info=None, hist_posi=None, path_values=None, is_debug=False)
    linewidth = 0.5
    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list['green']
        control_points = path.control_points
        color = Para.PATH_COLOR
        for i, (path_x, path_y, _, _) in enumerate(path_list):
            ax.plot(path_x[600:-630], path_y[600:-630], color=color[i], linewidth=linewidth)
        for i, four_points in enumerate(control_points):
            for point in four_points:
                ax.scatter(point[0], point[1], color=color[i], s=5, alpha=0.7, linewidth=linewidth)
            ax.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--',
                    color=color[i], alpha=0.5, linewidth=linewidth)
            ax.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--',
                    color=color[i], alpha=0.5, linewidth=linewidth)
            ax.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--',
                    color=color[i], alpha=0.5, linewidth=linewidth)
    plt.savefig('/home/yang/Desktop/毕业论文/图/chapter5/静态路径规划.pdf')
    plt.show()
    plt.close(f)


if __name__ == "__main__":
    plot_map()