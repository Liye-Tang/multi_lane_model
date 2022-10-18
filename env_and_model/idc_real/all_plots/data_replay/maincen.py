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

    def output(self):
        return dict(ego_info=self.ego_info,
                    other_info=self.other_info,
                    obs_info=self.obs_info,
                    decision_info=self.decision_info,
                    processed_info=self.info)

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

def main():
    data_dir = '/home/guanyang/下载/good_cases'
    # replay_cases = [13]
    df_list = []
    dict2save = {}
    for i in range(1, 33):
        case_dir = data_dir + '/case' + str(i) + '/best_try_1'  # use best try 1
        all_dir_in_case_dir = os.listdir(case_dir)
        for dir_in_case_dir in all_dir_in_case_dir:
            if dir_in_case_dir.startswith('try'):
                try_dir = case_dir + '/' + dir_in_case_dir
        get_param(i)
        replay_data = get_replay_data(try_dir, start_time=0)
        data_replay = DataReplay(replay_data, try_dir, replay_speed=1, case_idx=i)
        dict2save.update({i: data_replay.output()})
        # print(dict2save)
    np.save('./dict2save.npy', dict2save)


if __name__ == '__main__':
    main()
