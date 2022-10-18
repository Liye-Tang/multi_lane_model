# -*- coding: utf-8 -*-
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from env_and_model.idc_real.dynamics_and_models import ReferencePath
from env_and_model.idc_real.endtoend_env_utils import *
from env_and_model.idc_real.all_plots.data_replay.render_utils import *

# config = {
#     "font.family": 'serif',  # 衬线字体
#     "font.size": 9,
#     "font.serif": ['SimSun'],  # 宋体
#     "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#     'axes.unicode_minus': False  # 处理负号，即-号
# }

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


class DataReplay(object):
    def __init__(self, idc_planner_info_list, try_dir, replay_speed, case_idx):
        self.case_idx = case_idx
        # replay_cases = [13, 16, 18, 21, 24, 26, 31]
        self.case2totalstep = {13: 355, 16: None, 18: 235, 21: None, 24: None, 26: 265, 31: 320}
        self.steps2keep = {13: ['010', '095', '130', '200', '250', '270', '300', '350'],
                           16: ['015', '060', '120', '180'],
                           18: ['005', '030', '100', '220'],
                           21: ['005', '030', '130', '200'],
                           24: ['005', '030', '080', '090', '100', '120', '150', '270'],
                           26: ['015', '060', '080', '100', '120', '160', '190', '250'],
                           31: ['010', '080', '115', '135', '200', '240', '270', '315']}

        self.try_dir = try_dir
        self.info_list = idc_planner_info_list
        self.replay_speed = replay_speed
        self.task = None
        self.traffic_light = None
        self.light_phase = update_light_phase(case_idx, 0)
        self.ref_path = None
        self.total_step = None
        self.info = []
        self.ego_info = []
        self.other_info = []
        self.obs_info = []
        self.path_info = []
        self.decision_info = []
        self.plot_dict = dict(v_x=[],  # 1
                              phi=[],  # 2
                              delta_y=[],  # 3
                              delta_phi=[],
                              r=[],  # 4
                              delta_v=[],  # 5
                              acc_cmd=[],
                              acc_real=[],  # 6
                              steer_cmd=[],
                              steer_real=[],  # 7
                              pure_dctime_ms=[],  # 8
                              path_idx=[],
                              is_safe=[]  # 9
                              )
        self.get_info()
        self.interested_info = get_list_of_participants_in_obs(self.info_list)

    def get_info(self):
        self.total_step = self.case2totalstep.get(self.case_idx)
        if self.total_step is None:
            self.total_step = len(self.info_list)
        self.task = self.info_list[0]['task']
        self.traffic_light = self.info_list[0]['traffic_light']
        self.ref_path = ReferencePath(self.task)
        self.plot_dict['path_value'] = get_list_of_path_values(self.info_list)
        for i in range(self.total_step):
            info_dict = self.info_list[i]
            # ego state
            for _ in ['v_x', 'v_y', 'r']:
                if _ not in info_dict['ego_state'].keys():
                    info_dict['ego_state'][_] = 0
                    self.info_list[i]['ego_state'][_] = 0
            self.ego_info.append(info_dict['ego_state'])

            # other state
            try:
                for other_state in info_dict['other_state']:
                    other_state['phi'] *= 180 / pi
                self.other_info.append(info_dict['other_state'])
            except:
                self.other_info.append([])

            # decision info
            # add 0
            for _ in ['selected_path_idx', 'is_safe', 'normalized_front_wheel_clamp']:
                if _ not in info_dict['decision'].keys():
                    info_dict['decision'][_] = 0
                    self.info_list[i]['decision'][_] = 0
            self.decision_info.append(info_dict['decision'])

            # obs info
            obs_info_dict = {}
            obs_dim_list = [Para.EGO_ENCODING_DIM, Para.TRACK_ENCODING_DIM, Para.LIGHT_ENCODING_DIM,
                            Para.TASK_ENCODING_DIM, Para.REF_ENCODING_DIM, Para.HIS_ACT_ENCODING_DIM]

            selected_path_idx = info_dict['decision']['selected_path_idx']
            obs_info_dict['obs_vector'] = np.array(info_dict['obs_vector'][selected_path_idx]['input_vector'])
            obs_info_dict['ego_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][:obs_dim_list[0]]
            obs_info_dict['track_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][
                                          obs_dim_list[0]:sum(obs_dim_list[:2])]
            obs_info_dict['light_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][
                                          sum(obs_dim_list[:2]):sum(obs_dim_list[:3])]
            obs_info_dict['task_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][
                                         sum(obs_dim_list[:3]):sum(obs_dim_list[:4])]
            obs_info_dict['ref_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][
                                        sum(obs_dim_list[:4]):sum(obs_dim_list[:5])]
            obs_info_dict['his_act_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][
                                            sum(obs_dim_list[:5]):sum(obs_dim_list[:6])]
            self.obs_info.append(obs_info_dict)

            # process info
            processed_info_dict = {}
            processed_info_dict['traj_pose'] = info_dict['traj_pose']
            processed_info_dict['attn_vector'] = info_dict['attn_vector'][0]['input_vector']
            self.info.append(processed_info_dict)

            # plot_dict
            self.plot_dict['v_x'].append(info_dict['ego_state']['v_x'])
            ego_phi = info_dict['ego_state']['phi'] * 180 / pi
            if ego_phi < -100:
                ego_phi += 360
            self.plot_dict['phi'].append(ego_phi)  # deg
            self.plot_dict['delta_y'].append(obs_info_dict['track_info'][0])
            self.plot_dict['delta_phi'].append(obs_info_dict['track_info'][1])  # deg
            self.plot_dict['r'].append(info_dict['ego_state']['r'] * 180 / pi)  # deg/s
            self.plot_dict['delta_v'].append(obs_info_dict['track_info'][2])
            self.plot_dict['acc_cmd'].append(info_dict['decision']['normalized_acc_clamp'] * ACC_SCALE - ACC_SHIFT)
            self.plot_dict['acc_real'].append(
                info_dict['traj_pose'][-1]['y'] if 'y' in info_dict['traj_pose'][-1].keys() else 0)
            self.plot_dict['steer_cmd'].append(
                info_dict['decision']['normalized_front_wheel_clamp'] * STEER_SCALE * STEER_RATIO * 180 / pi)
            self.plot_dict['steer_real'].append(
                info_dict['traj_pose'][-1]['x'] if 'x' in info_dict['traj_pose'][-1].keys() else 0)
            decision_time_ms = info_dict['decision']['decision_time_ns']
            safety_shield_time_ms = info_dict['decision']['safety_shield_time_ns']
            path_num = 2 if self.case_idx in [2, 8, 9, 10, 11, 22, 23, 24, 25] else 3
            pure_dctime_ms = (decision_time_ms - safety_shield_time_ms) / (1e6 * path_num)
            self.plot_dict['pure_dctime_ms'].append(pure_dctime_ms)
            self.plot_dict['path_idx'].append(info_dict['decision']['selected_path_idx'])
            self.plot_dict['is_safe'].append(info_dict['decision']['is_safe'])

    def replay(self, save_video=False):
        plt.ion()
        fig_path = os.path.join(self.try_dir, 'replay_results_new', 'figs')
        if save_video:
            if os.path.exists(fig_path):
                shutil.rmtree(fig_path)
                os.makedirs(fig_path)
            else:
                os.makedirs(fig_path)
        hist_posi = []
        f = plt.figure(figsize=(pt2inch(420 / 4), pt2inch(420 / 4)), dpi=300)
        for i in range(0, self.total_step):
            self.light_phase = update_light_phase(self.case_idx, i/10)
            hist_posi.append((self.ego_info[i]['x'], self.ego_info[i]['y']))
            if i % self.replay_speed == 0:
                self.ref_path.set_path(self.traffic_light, self.decision_info[i]['selected_path_idx'])
                self.plot_for_replay(self.obs_info[i]['obs_vector'], self.ref_path, self.interested_info[i],
                                     self.other_info[i], hist_posi, self.info[i]['attn_vector'], save_video,
                                     i, postfix='pdf')
        plt.close(f)
        for i, step in enumerate(self.steps2keep[self.case_idx]):
            mycopyfile(fig_path + '/' + step + '.pdf', '/home/yang/Desktop/毕业论文/图/chapter5/场景/{}/决控过程/origin/'.format(self.case_idx))
            mycopyfile(fig_path + '/' + step + '.pdf', '/home/yang/Desktop/毕业论文/图/chapter5/场景/{}/决控过程/'.format(self.case_idx), '{}.pdf'.format(i+1))

        if save_video:
            subprocess.call(['ffmpeg', '-framerate', '10', '-i', fig_path + '/' + '%03d.png', '-c:v', 'copy',
                             self.try_dir + '/replay_results_new' + '/video.mp4'])
            mycopyfile(self.try_dir + '/replay_results_new' + '/video.mp4', '/home/yang/Desktop/毕业论文/图/chapter5/场景/{}/决控过程/'.format(self.case_idx))


    def plot_for_replay(self, obs, ref_path, interested_other, other_info, hist_posi, attn_weights, save_video,
                        replay_counter, postfix='pdf'):
        render(light_phase=self.light_phase, all_other=other_info,
               interested_other=interested_other, attn_weights=np.array(attn_weights),
               obs=obs, ref_path=ref_path,
               future_n_point=None, action=None, done_type=None,
               reward_info=None, hist_posi=hist_posi, path_values=None, is_debug=False)
        if save_video:
            fig_path = os.path.join(self.try_dir, 'replay_results_new', 'figs')
            plt.show()
            plt.pause(0.001)
            plt.savefig(fig_path + '/{:03d}.'.format(int(replay_counter)) + postfix,
                        bbox_inches='tight', pad_inches=0, figsize=(pt2inch(420 / 2), pt2inch(420 / 2)), dpi=300)
        else:
            # plt.show()
            plt.pause(0.001)

    def featured(self):
        # scalar
        total_time = 0.1 * self.total_step
        path_switch_num = sum([1 if self.plot_dict['path_idx'][i] != self.plot_dict['path_idx'][i - 1] else 0
                               for i in range(1, self.total_step)])
        ss_num = sum([1 if not self.plot_dict['is_safe'][i] and self.plot_dict['is_safe'][i - 1] else 0
                      for i in range(1, self.total_step)])

        # list
        dc_time_list = np.array(self.plot_dict['pure_dctime_ms'])
        delta_y_list = np.array(self.plot_dict['delta_y'])
        delta_phi_list = np.array(self.plot_dict['delta_phi'])
        delta_v_list = np.array(self.plot_dict['delta_v'])
        v_list = np.array(self.plot_dict['v_x'])
        r_list = np.array(self.plot_dict['r'])
        steer_list = np.array(self.plot_dict['steer_real'])
        acc_list = np.array(self.plot_dict['acc_real'])
        return dict(case_idx=self.case_idx,
                    total_time=total_time,
                    path_switch_num=path_switch_num,
                    ss_num=ss_num,
                    dc_time_list=dc_time_list,
                    delta_y_list=delta_y_list,
                    delta_phi_list=delta_phi_list,
                    delta_v_list=delta_v_list,
                    v_list=v_list,
                    r_list=r_list,
                    steer_list=steer_list,
                    acc_list=acc_list)

    def plot_curve(self):
        curve_path = os.path.join(self.try_dir, 'replay_results_new', 'curves')
        if os.path.exists(curve_path):
            shutil.rmtree(curve_path)
            os.makedirs(curve_path)
        else:
            os.makedirs(curve_path)
        time_line = np.array([0.1 * k for k in range(self.total_step)])
        color1, color2, color3, color4 = 'k', 'r', 'b', 'g'
        style1, style2, style3, style4 = '-', '--', '-.', ':'
        linewidth = 0.5
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

        axes_list = [0.3, 0.3, 0.68, 0.5]
        for key, value in self.plot_dict.items():
            f_name = None
            f = plt.figure(figsize=(pt2inch(420 / 3), pt2inch(420 / 3 * (2 / 3))), dpi=200)
            if key == 'v_x':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('速度($\mathrm{m/s}$)', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                f_name = '速度'
            elif key == 'phi':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('航向角(($\degree$))', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                f_name = '航向角'
            elif key == 'delta_y':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('位置误差($\mathrm{m}$)', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                f_name = '位置误差'
            elif key == 'delta_phi':
                pass
            elif key == 'r':
                ax = plt.axes(axes_list)
                ax.plot(time_line, self.plot_dict['delta_phi'], linewidth=linewidth, color=color1, linestyle=style1,
                        label='航向角误差')
                ax.plot(time_line, value, linewidth=linewidth, color=color2, linestyle=style2,
                        label='横摆角速度')
                ax.set_ylabel('角度(($\degree$))/\n角速度(($\degree$)$\mathrm{/s}$)', fontproperties=SimSun)
                ax.yaxis.set_label_coords(-0.2, 0.4)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                ax.legend(ncol=2, frameon=False, prop={'family': 'SimSun', 'size': 9},
                          bbox_to_anchor=(1.05, 1.5), columnspacing=0.4, handletextpad=0.1,
                          handlelength=1)
                f_name = '角度和角速度'
            elif key == 'delta_v':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('速度误差($\mathrm{m/s}$)', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                f_name = '速度误差'
            elif key == 'acc_cmd':
                pass
            elif key == 'acc_real':
                ax = plt.axes(axes_list)
                ax.plot(time_line, self.plot_dict['acc_cmd'], linewidth=linewidth, color=color1, linestyle=style1,
                        label='期望')
                ax.plot(time_line, value, linewidth=linewidth, color=color2, linestyle=style2,
                        label='实际')
                ax.set_ylabel('加速度($\mathrm{m/s^2}$)', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                ax.legend(ncol=2, frameon=False, prop={'family': 'SimSun', 'size': 9},
                          bbox_to_anchor=(1.05, 1.5))
                f_name = '加速度'
            elif key == 'steer_cmd':
                pass
            elif key == 'steer_real':
                ax = plt.axes(axes_list)
                ax.plot(time_line, self.plot_dict['steer_cmd'], linewidth=linewidth, color=color1, linestyle=style1,
                        label='期望')
                ax.plot(time_line, value, linewidth=linewidth, color=color2, linestyle=style2,
                        label='实际')
                ax.set_ylabel('方向盘转角(($\degree$))', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                ax.legend(ncol=2, frameon=False, prop={'family': 'SimSun', 'size': 9},
                          bbox_to_anchor=(1.05, 1.5))
                f_name = '方向盘转角'
            elif key == 'pure_dctime_ms':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('计算时间($\mathrm{ms}$)', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                f_name = '计算时间'
            elif key == 'path_idx':
                ax = plt.axes(axes_list)
                ax.plot(time_line, value, linewidth=linewidth, color=color1, linestyle=style1)
                ax.set_ylabel('路径编号', fontproperties=SimSun)
                ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
                f_name = '路径选择'
            # elif key == 'is_safe':
            #     ax = plt.axes(axes_list)
            #     ax.plot(time_line, self.plot_dict['path_idx'], linewidth=linewidth, color=color1,
            #             linestyle=style1, label='路径')
            #     ax.plot(time_line, value, linewidth=linewidth, color=color2,
            #             linestyle=style2, label='护盾')
            #     ax.set_ylabel('路径/护盾', fontproperties=SimSun)
            #     ax.set_xlabel("时间($\mathrm{s}$)", fontproperties=SimSun)
            #     ax.legend(ncol=2, frameon=False, prop={'family': 'SimSun', 'size': 9},
            #               bbox_to_anchor=(1.05, 1.5))
            #     f_name = '路径和安全护盾'
            plt.savefig(curve_path + '/{}.pdf'.format(f_name))
            plt.close(f)
        for curve in os.listdir(curve_path):
            mycopyfile(curve_path + '/' + curve, '/home/yang/Desktop/毕业论文/图/chapter5/场景/{}/曲线/'.format(self.case_idx))


def get_param(i):
    global ACC_SCALE
    global ACC_SHIFT
    global STEER_SCALE
    global STEER_SHIFT
    if i > 12:
        ACC_SCALE = 1.5
        ACC_SHIFT = 0.5
        STEER_SCALE = 0.3
        STEER_SHIFT = 0
    else:
        ACC_SCALE = 2.25
        ACC_SHIFT = 0.75
        STEER_SCALE = 0.4
        STEER_SHIFT = 0


def deal_with_total_df(total_df, data_dir):
    # calculate statistics
    # pd.DataFrame(dict(case_idx=self.case_idx,
    #                   total_time=total_time,
    #                   path_switch_num=path_switch_num,
    #                   ss_num=ss_num,
    #                   a_w_list=a_w_list,
    #                   dc_time_list=dc_time_list,
    #                   delta_y_list=delta_y_list,
    #                   delta_phi_list=delta_phi_list,
    #                   delta_v_list=delta_v_list,
    #                   v_list=v_list,
    #                   r_list=r_list,
    #                   steer_list=steer_list,
    #                   acc_list=acc_list))
    I_time_mean = total_df['dc_time_list'].mean()
    I_time_std = total_df['dc_time_list'].std()
    I_comfort = np.sqrt(total_df['acc_list'].map(lambda x: x ** 2).mean())
    non_green_cases = [16, 17, 18, 19, 20, 21]
    green_case_idx = total_df[(total_df["case_idx"] <= 16) & (total_df["case_idx"] <= 21)].index
    green_df = total_df.drop(green_case_idx)
    I_efficiency_mean = green_df['total_time'].mean()
    I_efficiency_std = green_df['total_time'].std()
    statis = dict(I_time_mean=I_time_mean, I_time_std=I_time_std, I_comfort=I_comfort,
                  I_efficiency_mean=I_efficiency_mean, I_efficiency_std=I_efficiency_std)
    print(statis)
    # plot figure
    bar_path = os.path.join(data_dir, 'barplots')
    if os.path.exists(bar_path):
        shutil.rmtree(bar_path)
        os.makedirs(bar_path)
    else:
        os.makedirs(bar_path)
    width_pt, height_pt = 420, 70
    axes_list = [0.1, 0.4, 0.88, 0.4]
    f = plt.figure(figsize=(pt2inch(width_pt), pt2inch(height_pt)), dpi=200)
    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="total_time", data=total_df, ax=ax)
    ax.set_ylabel('通行时间($\mathrm{s}$)', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('通行时间'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="dc_time_list", data=total_df, ax=ax)
    ax.set_ylabel('计算时间($\mathrm{ms}$)', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('计算时间'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="delta_y_list", data=total_df, ax=ax)
    ax.set_ylabel('位置误差($\mathrm{m}$)', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('位置误差'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="delta_v_list", data=total_df, ax=ax)
    ax.set_ylabel('速度误差($\mathrm{m/s}$)', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('速度误差'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="delta_phi_list", data=total_df, ax=ax)
    ax.set_ylabel('航向角误差(($\degree$))', fontproperties=SimSun)
    ax.yaxis.set_label_coords(-0.07, 0.25)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('航向角误差'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="v_list", data=total_df, ax=ax)
    ax.set_ylabel('速度($\mathrm{m/s}$)', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('速度'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="r_list", data=total_df, ax=ax)
    ax.set_ylabel('横摆角速度(($\degree$)$\mathrm{/s}$)', fontproperties=SimSun)
    ax.yaxis.set_label_coords(-0.07, 0.28)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('横摆角速度'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="steer_list", data=total_df, ax=ax)
    ax.set_ylabel('方向盘转角(($\degree$))', fontproperties=SimSun)
    ax.yaxis.set_label_coords(-0.07, 0.25)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('方向盘转角'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="acc_list", data=total_df, ax=ax)
    ax.set_ylabel('加速度($\mathrm{m/s^2}$)', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('加速度'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="path_switch_num", data=total_df, ax=ax)
    ax.set_ylabel('路径切换次数', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('路径切换次数'))
    plt.clf()

    ax = plt.axes(axes_list)
    sns.barplot(x="case_idx", y="ss_num", data=total_df, ax=ax)
    ax.set_ylabel('护盾启动次数', fontproperties=SimSun)
    ax.set_xlabel("场景编号", fontproperties=SimSun)
    plt.savefig(bar_path + '/{}.pdf'.format('护盾启动次数'))
    plt.close(f)
    for bar in os.listdir(bar_path):
        mycopyfile(bar_path + '/' + bar, '/home/yang/Desktop/毕业论文/图/chapter5/场景/统计/')


def main():
    data_dir = '/home/yang/Desktop/毕业论文/实车实验数据/good_cases_new/good_cases'
    replay_cases = [13, 16, 18, 21, 24, 26, 31]
    # replay_cases = [13]
    df_list = []
    for i in range(1, 33):
        case_dir = data_dir + '/case' + str(i) + '/best_try_1'  # use best try 1
        all_dir_in_case_dir = os.listdir(case_dir)
        for dir_in_case_dir in all_dir_in_case_dir:
            if dir_in_case_dir.startswith('try'):
                try_dir = case_dir + '/' + dir_in_case_dir
        get_param(i)
        replay_data = get_replay_data(try_dir, start_time=0)
        data_replay = DataReplay(replay_data, try_dir, replay_speed=1, case_idx=i)
        if i in replay_cases:
            pass
            # data_replay.replay(save_video=True)
            # data_replay.plot_curve()
        df_list.append(pd.DataFrame(data_replay.featured()))
    total_df = df_list[0].append(df_list[1:], ignore_index=True)
    deal_with_total_df(total_df, data_dir)


if __name__ == '__main__':
    main()
