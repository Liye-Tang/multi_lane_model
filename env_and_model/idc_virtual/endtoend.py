#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================
import warnings
from math import sqrt

import gym

from env_and_model.idc_virtual.dynamics_and_models import VehicleDynamics, ReferencePath, IdcVirtualModel
from env_and_model.idc_virtual.endtoend_env_utils import *
from env_and_model.idc_virtual.traffic import Traffic
from env_and_model.idc_virtual.utils.sensor import Perception

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class IdcVirtualEnv(gym.Env):
    def __init__(self,
                 mode='training',
                 multi_display=False,
                 **kwargs):
        self.mode = mode
        self.dynamics = VehicleDynamics(if_model=False)
        self.interested_other = None
        self.detected_vehicles = None
        self.all_other = None
        self.all_detected_other = None
        self.ego_dynamics = None
        self.init_state = {}
        self.ego_l, self.ego_w = Para.L, Para.W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.light_phase = None
        self.light_encoding = None
        self.task_encoding = None
        self.step_length = 100  # ms
        self.step_time = self.step_length / 1000.0
        self.obs = None
        self.action = None

        self.done_type = 'not_done_yet'
        self.reward_dict = None
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.ref_info_dim = Para.REF_ENCODING_DIM
        self.his_act_info_dim = Para.HIS_ACT_ENCODING_DIM
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.light_info_dim,
                                    self.task_info_dim, self.ref_info_dim, self.his_act_info_dim])
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])

        self.veh_mode_dict = None
        self.bicycle_mode_dict = None
        self.person_mode_dict = None
        self.training_task = None
        self.env_model = None
        self.ref_path = None
        self.future_n_point = None
        self.future_point_num = Para.FUTURE_POINT_NUM
        self.step_limit = 200
        self.curr_step = 0
        self.obs_scale = Para.OBS_SCALE
        self.rew_scale, self.rew_shift = 0.1, 0.
        self.punish_scale = 0.1
        self.perception = Perception()

        self.vector_noise = False
        # if self.vector_noise:
        #     self.rng = np.random.default_rng(12345)
        self.action_store = ActionStore(maxlen=2)

        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state)
            self.reset()
            observation, _reward, done, _info = self.step(np.array([0.,0.]))
            self._set_observation_space(observation)
            plt.ion()

    def reset(self, **kwargs):  # kwargs include three keys
        green_prob = kwargs['green_prob'] if 'green_prob' in kwargs else 1
        self.light_phase = self.traffic.init_light(green_prob)
        rnd = random.random()
        self.training_task = 'left' if rnd < 0.3333 else ('straight' if rnd < 0.6666 else 'right')
        self.task_encoding = TASK_ENCODING[self.training_task]
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        self.ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.bicycle_mode_dict = BIKE_MODE_DICT[self.training_task]
        self.person_mode_dict = PERSON_MODE_DICT[self.training_task]
        self.env_model = IdcVirtualModel()
        self.action_store.reset()
        self.init_state = self._reset_init_state(LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )
        self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point = self._get_obs()
        self.action = None
        self.reward_dict = None
        self.done_type = 'not_done_yet'
        all_info = dict(future_n_point=self.future_n_point, mask=other_mask_vector)
        self.curr_step = 0
        return self.obs, all_info

    def close(self):
        del self.traffic

    def step(self, action):
        self.action_store.put(action)
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_dict = self._compute_reward(self.obs, self.action, action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point = self._get_obs()
        self.done_type, done = self._judge_done()
        self.curr_step += 1
        if self.done_type in ['collision', 'break_road_constrain']:
            self.reward_dict['punish'] += 50
        all_info.update(
            {'reward_dict': self.reward_dict, 'future_n_point': self.future_n_point, 'mask': other_mask_vector})
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   l=self.ego_l,
                   w=self.ego_w,
                   alpha_f=next_ego_params[0],
                   alpha_r=next_ego_params[1],
                   miu_f=next_ego_params[2],
                   miu_r=next_ego_params[3], )
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = abs(3 * miu_f * F_zf / C_f), abs(3 * miu_r * F_zr / C_r)
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x']) + 1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

        corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        corner_point=corner_point))
        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_other = self.traffic.n_ego_others['ego']  # coordination 2
        self.perception.get_ego(ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'])
        self.all_detected_other = self.perception.process(self.all_other)
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.light_phase = self.traffic.light_phase

        all_info = dict(all_other=self.all_other,
                        all_detected_other=self.all_detected_other,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.light_phase)
        return all_info

    def _judge_done(self):
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_speed_limit():
            return 'break_speed_limit', 1
        # elif self._break_stability():
        #     return 'break_stability', 1
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self.curr_step > self.step_limit:
            return 'exceed_step_limit', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_lateral, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        return True if abs(delta_lateral) > 15 else False

    def _break_speed_limit(self):
        v = self.obs[0]
        return True if v > 10 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        # TODO(guanyang): check the correctness
        return True if self.light_phase != 0 and \
                       self.light_phase != 1 and \
                       self.ego_dynamics['y'] > -Para.CROSSROAD_SIZE / 2 and \
                       self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -Para.CROSSROAD_SIZE / 2 - 20 and 0 < y < Para.LANE_NUMBER * Para.LANE_WIDTH else False
        elif self.training_task == 'right':
            return True if x > Para.CROSSROAD_SIZE / 2 + 20 and -Para.LANE_NUMBER * Para.LANE_WIDTH < y < 0 else False
        else:
            assert self.training_task == 'straight'
            return True if y > Para.CROSSROAD_SIZE / 2 + 20 and 0 < x < Para.LANE_NUMBER * Para.LANE_WIDTH else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = Para.STEER_SCALE * steer_norm
        if self.obs[4] < -Para.CROSSROAD_SIZE/2:
            scaled_steer = 0.
        scaled_a_x = Para.ACC_SCALE * a_x_norm + Para.ACC_SHIFT
        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)
        next_ego_state, next_ego_params = next_ego_state.numpy()[0], next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        other_vector, other_mask_vector = self._construct_other_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        track_vector = self.ref_path.tracking_error_vector_vectorized(ego_vector[3], ego_vector[4], ego_vector[5],
                                                                      ego_vector[0])  # 3 for x; 4 foy y
        future_n_point = self.ref_path.get_future_n_point(ego_vector[3], ego_vector[4], self.future_point_num * 2)
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        vector = np.concatenate((ego_vector, track_vector, self.light_encoding, self.task_encoding,
                                 self.ref_path.ref_encoding, self.action_store[0], self.action_store[1], other_vector),
                                axis=0)
        vector = vector.astype(np.float32)
        vector = convert_to_rela(vector)

        return vector, other_mask_vector, future_n_point

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        return np.array(ego_feature, dtype=np.float32)

    def _construct_other_vector_short(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        other_vector = []
        other_mask_vector = []

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_other(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            du_b, dr_b, rl_b, ru_b, ud_b, ul_b, lr_b, ld_b = [], [], [], [], [], [], [], []
            c0, c1, c2, c3 = [], [], [], []

            other_padding = dict(type="none",
                                 x=7.631,
                                 y=-64.6, v=0.,
                                 phi=0., w=0.1, l=0.1, route=('1o', '3i'), partici_type=[0., 0., 0.], turn_rad=0.,
                                 exist=False)  # l=0.1, w=0.1 is for em planner

            def cal_turn_rad(v):
                if v['type'] in ['bicycle_1', 'bicycle_2', 'bicycle_3', 'DEFAULT_PEDTYPE']:
                    return 0.
                if not (-Para.CROSSROAD_SIZE / 2 < v['x'] < Para.CROSSROAD_SIZE / 2 and
                        -Para.CROSSROAD_SIZE / 2 < v['y'] < Para.CROSSROAD_SIZE / 2):
                    turn_rad = 0.
                else:
                    start = v['route'][0]
                    end = v['route'][1]
                    if (start == name_setting['do'] and end == name_setting['ui']) or \
                            (start == name_setting['ro'] and end == name_setting['li']) or \
                            (start == name_setting['uo'] and end == name_setting['di']) or \
                            (start == name_setting['lo'] and end == name_setting['ri']):
                        turn_rad = 0.
                    elif (start == name_setting['do'] and end == name_setting['ri']) or \
                            (start == name_setting['ro'] and end == name_setting['ui']) or \
                            (start == name_setting['uo'] and end == name_setting['li']) or \
                            (start == name_setting['lo'] and end == name_setting['di']):
                        turn_rad = -1 / (Para.CROSSROAD_SIZE / 2 - 2.5 * Para.LANE_WIDTH)
                    elif (start == name_setting['do'] and end == name_setting['li']) or \
                            (start == name_setting['ro'] and end == name_setting['di']) or \
                            (start == name_setting['uo'] and end == name_setting['ri']) or \
                            (start == name_setting['lo'] and end == name_setting['ui']):
                        turn_rad = 1 / (Para.CROSSROAD_SIZE / 2 + 0.5 * Para.LANE_WIDTH)
                return turn_rad

            for v in vs:
                if v['type'] in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    v.update(partici_type=[0., 1., 0.], turn_rad=0.0, exist=True)
                    route_list = v['route']
                    start = route_list[0]
                    end = route_list[1]

                    if start == name_setting['do'] and end == name_setting['ui']:
                        du_b.append(v)
                    elif start == name_setting['do'] and end == name_setting['ri']:
                        dr_b.append(v)

                    elif start == name_setting['ro'] and end == name_setting['li']:
                        rl_b.append(v)
                    elif start == name_setting['ro'] and end == name_setting['ui']:
                        ru_b.append(v)

                    elif start == name_setting['uo'] and end == name_setting['di']:
                        ud_b.append(v)
                    elif start == name_setting['uo'] and end == name_setting['li']:
                        ul_b.append(v)

                    elif start == name_setting['lo'] and end == name_setting['ri']:
                        lr_b.append(v)
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        ld_b.append(v)

                elif v['type'] == 'DEFAULT_PEDTYPE':
                    v.update(partici_type=[0., 1., 0.], turn_rad=0.0, exist=True)
                    if -14.0 < v['x'] < 14.0 and 21.0 < v['y'] < 25.0:
                        c0.append(v)
                    elif 21.0 < v['x'] < 25.0 and -14.0 < v['y'] < 14.0:
                        c1.append(v)
                    elif -14.0 < v['x'] < 14.0 and -25.0 < v['y'] < -21.0:
                        c2.append(v)
                    elif -25.0 < v['x'] < -21.0 and -14.0 < v['y'] < 14.0:
                        c3.append(v)
                else:
                    v.update(partici_type=[0., 0., 1.], turn_rad=cal_turn_rad(v), exist=True)
                    route_list = v['route']
                    start = route_list[0]
                    end = route_list[1]
                    if start == name_setting['do'] and end == name_setting['li']:
                        dl.append(v)
                    elif start == name_setting['do'] and end == name_setting['ui']:
                        du.append(v)
                    elif start == name_setting['do'] and end == name_setting['ri']:
                        dr.append(v)

                    elif start == name_setting['ro'] and end == name_setting['di']:
                        rd.append(v)
                    elif start == name_setting['ro'] and end == name_setting['li']:
                        rl.append(v)
                    elif start == name_setting['ro'] and end == name_setting['ui']:
                        ru.append(v)

                    elif start == name_setting['uo'] and end == name_setting['ri']:
                        ur.append(v)
                    elif start == name_setting['uo'] and end == name_setting['di']:
                        ud.append(v)
                    elif start == name_setting['uo'] and end == name_setting['li']:
                        ul.append(v)

                    elif start == name_setting['lo'] and end == name_setting['ui']:
                        lu.append(v)
                    elif start == name_setting['lo'] and end == name_setting['ri']:
                        lr.append(v)
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        ld.append(v)

            # fetch bicycle in range
            if task == 'straight':
                du_b = list(filter(lambda v: ego_y - 6 < v['y'] < Para.CROSSROAD_SIZE / 2 and v['x'] < ego_x + 8, du_b))
            elif task == 'right':
                du_b = list(filter(
                    lambda v: max(-Para.LANE_NUMBER * Para.LANE_WIDTH, ego_y - 7) < v['y'] < min(0, ego_y + 3) and
                              v['x'] < ego_x + 8, du_b))
            ud_b = list(filter(
                lambda v: max(ego_y - 3, -Para.CROSSROAD_SIZE / 2) < v['y'] < min(Para.CROSSROAD_SIZE / 2,
                                                                                  ego_y + 7) and
                          ego_x > v['x'] > ego_x - 25, ud_b))  # interest of left
            lr_b = list(
                filter(lambda v: 0 < v['x'] < min(Para.CROSSROAD_SIZE / 2 + 40, ego_x + 10), lr_b))  # interest of right
            tmp_bike = []
            for mode, _ in BIKE_MODE_DICT[task].items():
                tmp_bike.extend(eval(mode))
            while len(tmp_bike) < self.bike_num:
                tmp_bike.append(other_padding)
            if len(tmp_bike) > self.bike_num:
                tmp_bike = sorted(tmp_bike, key=lambda v: sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2))
                tmp_bike = tmp_bike[:self.bike_num]

            # fetch person in range
            c1_du = list(filter(
                lambda v: (v['phi'] > 0) and
                          max(-Para.LANE_NUMBER * Para.LANE_WIDTH, ego_y - 7) < v['y'] < min(0, ego_y + 3) and
                          ego_x + 20 > v['x'] > ego_x - Para.L, c1))  # interest of right
            c1_ud = list(filter(
                lambda v: (v['phi'] < 0) and
                          max(-Para.LANE_NUMBER * Para.LANE_WIDTH, ego_y - 3) < v['y'] < min(0, ego_y + 7) and
                          ego_x + 20 > v['x'] > ego_x - Para.L, c1))  # interest of right
            c1 = c1_du + c1_ud
            c2 = list(filter(lambda v: max(0, ego_x - 5) < v['x'] and v['y'] > ego_y - Para.L, c2))  # interest of right
            c3_du = list(filter(
                lambda v: (v['phi'] > 0) and
                          max(0, ego_y - 7) < v['y'] < min(ego_y + 3, Para.LANE_NUMBER * Para.LANE_WIDTH) and
                          ego_x - 25 < v['x'] < ego_x + Para.L, c3))  # interest of left
            c3_ud = list(filter(
                lambda v: (v['phi'] < 0) and
                          max(0, ego_y - 3) < v['y'] < min(ego_y + 7, Para.CROSSROAD_SIZE / 2) and
                          ego_x - 25 < v['x'] < ego_x + Para.L, c3))  # interest of left
            c3 = c3_du + c3_ud

            tmp_ped = []
            for mode, _ in PERSON_MODE_DICT[task].items():
                tmp_ped.extend(eval(mode))
            while len(tmp_ped) < self.person_num:
                tmp_ped.append(other_padding)
            if len(tmp_ped) > self.person_num:
                tmp_ped = sorted(tmp_ped, key=lambda v: sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2))
                tmp_ped = tmp_ped[:self.person_num]

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > max(ego_x - 25, -Para.CROSSROAD_SIZE / 2 - 40) and v['y'] > ego_y - 2,
                             dl))  # interest of left straight
            du = list(filter(lambda v: ego_y - 2 < v['y'] < Para.CROSSROAD_SIZE / 2 + 40 and v['x'] < ego_x + 5,
                             du))  # interest of left straight

            dr = list(filter(lambda v: v['x'] < min(ego_x + 25, Para.CROSSROAD_SIZE / 2 + 40) and v['y'] > ego_y - 8,
                             dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < Para.CROSSROAD_SIZE / 2 + 10 and v['y'] < Para.CROSSROAD_SIZE / 2 + 40,
                             ru))  # interest of straight

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < Para.CROSSROAD_SIZE / 2 + 10,
                                 ur))  # interest of straight
            elif task == 'right':
                ur = list(filter(lambda v: ego_x - 20 < v['x'] < min(ego_x + 25, Para.CROSSROAD_SIZE / 2 + 40) and v[
                    'y'] < Para.CROSSROAD_SIZE / 2, ur))  # interest of right
            ud = list(filter(
                lambda v: max(ego_y - 5, -Para.CROSSROAD_SIZE / 2) < v['y'] < Para.CROSSROAD_SIZE / 2 and ego_x + 5 > v[
                    'x'], ud))  # interest of left
            ul = list(filter(lambda v: max(-Para.CROSSROAD_SIZE / 2 - 40, ego_x - 25) < v['x'] < ego_x + 10 and v[
                'y'] < Para.CROSSROAD_SIZE / 2 + 10, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -Para.CROSSROAD_SIZE / 2 < v['x'] < Para.CROSSROAD_SIZE / 2 + 40,
                             lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            tmp_veh = []
            for mode, _ in VEHICLE_MODE_DICT[task].items():
                tmp_veh.extend(eval(mode))
            while len(tmp_veh) < self.veh_num:
                tmp_veh.append(other_padding)
            if len(tmp_veh) > self.veh_num:
                tmp_veh = sorted(tmp_veh, key=lambda v: sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2))
                tmp_veh = tmp_veh[:self.veh_num]

            return tmp_bike + tmp_ped + tmp_veh

        self.interested_other = filter_interested_other(self.all_detected_other, self.training_task)

        for other in self.interested_other:
            other_x, other_y, other_v, other_phi, other_l, other_w, other_type, other_turn_rad, other_mask = \
                other['x'], other['y'], other['v'], other['phi'], other['l'], other['w'], other['partici_type'], other[
                    'turn_rad'], other['exist']
            other_vector.extend(
                [other_x, other_y, other_v, other_phi, other_l, other_w] + other_type + [other_turn_rad])
            other_mask_vector.append(other_mask)

        return np.array(other_vector, dtype=np.float32), np.array(other_mask_vector, dtype=np.float32)

    def _reset_init_state(self, light_phase):
        front = False
        if self.training_task == 'left':
            if light_phase == 'green':
                random_index = np.random.randint(700, 1200 if front else 1200+1400/2)
            else:
                random_index = np.random.randint(700, 900)
        elif self.training_task == 'straight':
            if light_phase == 'green':
                random_index = np.random.randint(700, 1200 if front else 1200+1500/2)
            else:
                random_index = np.random.randint(700, 900)
        else:
            random_index = np.random.randint(700, 1200 if front else 1200+800/2)
        init_ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        x, y, phi, exp_v = init_ref_path.idx2point(random_index)
        v = exp_v * np.random.random()
        routeID = TASK2ROUTEID[self.training_task]
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x,
                             y=y,
                             phi=phi,
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def _compute_reward(self, obs, action, untransformed_action):
        obses, actions, untransformed_actions = obs[np.newaxis, :], action[np.newaxis, :], untransformed_action[np.newaxis, :]
        reward_dict = self.env_model.compute_rewards(obses, actions, untransformed_actions)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward_dict['rewards'], reward_dict

    def render(self, mode='human', attn_weights=None):
        render(light_phase=self.light_phase, all_other=self.all_other, detected_other=self.all_detected_other,
               interested_other=self.interested_other, attn_weights=attn_weights, obs=self.obs, ref_path=self.ref_path,
               future_n_point=self.future_n_point, action=self.action, done_type=self.done_type,
               reward_info=self.reward_dict, hist_posi=None, path_values=None,
               sensor_config=[(80., 360.), (100., 38.)], is_debug=True)
        plt.show()
        plt.pause(0.05)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    env = IdcVirtualEnv()
    env_model = IdcVirtualModel()
    obs, _ = env.reset()
    plt.ion()
    i = 0
    while i < 100000:
        for j in range(200):
            i += 1
            # action=2*np.random.random(2)-1
            action = np.array([0.5, 1], dtype=np.float32)

            # if obs[4] < -18:
            #     action = np.array([0, 1], dtype=np.float32)
            # elif obs[3] <= -18:
            #     action = np.array([0, 0], dtype=np.float32)
            # else:
            #     action = np.array([0.2, 0.33], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            # env_model.reset(obs[np.newaxis, :], info['future_n_point'][np.newaxis, :])
            # env_model.mode = 'training'
            # for _ in range(10):
            #     obs_model, reward_dict = env_model.rollout(action[np.newaxis, :])
            #     env_model.render()
            env.render()
            # if done:
            #     print(env.done_type)
            #     break
        obs, _ = env.reset()
        env.render()


if __name__ == '__main__':
    test_end2end()
