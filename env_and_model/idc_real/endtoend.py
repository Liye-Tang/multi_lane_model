#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from math import sqrt
from random import choice

import gym
from gym.utils import seeding
from numpy import sin, cos

from env_and_model.idc_real.dynamics_and_models import VehicleDynamics, ReferencePath, IdcRealModel
from env_and_model.idc_real.endtoend_env_utils import *
from env_and_model.idc_real.traffic import Traffic

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


class IdcRealEnv(gym.Env):
    def __init__(self,
                 mode='training',
                 multi_display=False,
                 traffic_mode='auto',  # 'auto' or 'user'
                 **kwargs):
        self.mode = mode
        self.traffic_mode = traffic_mode
        if traffic_mode == 'auto':
            self.traffic_case = None
        elif traffic_mode == 'user':
            self.traffic_case = choice(list(MODE2STEP.keys()))
        else:
            assert 1, 'setting wrong traffic mode'
        self.dynamics = VehicleDynamics(if_model=False)
        self.interested_other = None
        self.detected_vehicles = None
        self.all_other = None
        self.ego_dynamics = None
        self.init_state = {}
        self.ego_l, self.ego_w = Para.L, Para.W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.seed()
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
        # TODO(guanyang): determine these values
        self.rew_scale, self.rew_shift = 1., 0.
        self.punish_scale = 1.

        self.vector_noise = False
        if self.vector_noise:
            self.rng = np.random.default_rng(12345)
        self.action_store = ActionStore(maxlen=2)

        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state,
                                   traffic_mode=traffic_mode,
                                   traffic_case=self.traffic_case)
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        if self.traffic_mode == 'auto':
            self.traffic_case = None
        elif self.traffic_mode == 'user':
            self.traffic_case = choice(list(MODE2STEP.keys()))
            # self.traffic_case = 'green_mix_left_1'
        else:
            assert 1, 'setting wrong traffic mode'
        self.light_phase = self.traffic.init_light(self.traffic_case)
        if self.traffic_mode == 'auto':
            self.training_task = choice(['left', 'straight', 'right'])
        else:
            self.training_task = str(self.traffic_case).split('_')[-2]
        self.task_encoding = TASK_ENCODING[self.training_task]
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        self.ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.bicycle_mode_dict = BIKE_MODE_DICT[self.training_task]
        self.person_mode_dict = PERSON_MODE_DICT[self.training_task]
        self.env_model = IdcRealModel()
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
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.light_phase = self.traffic.light_phase

        all_info = dict(all_other=self.all_other,
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
        elif self._break_stability():
            return 'break_stability', 1
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
        x_trans, y_trans, _ = rotate_coordination(self.ego_dynamics['x'], self.ego_dynamics['y'], 0, Para.ANGLE_D-90)
        OFFSET_D_X_trans, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)
        return True if self.light_phase > 2 and y_trans > OFFSET_D_Y_trans and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -Para.CROSSROAD_SIZE_LAT / 2 - 30 and Para.OFFSET_L + Para.L_GREEN < y < Para.OFFSET_L + Para.L_GREEN +  Para.L_OUT_0 + Para.L_OUT_1+ Para.L_OUT_2 else False
        elif self.training_task == 'right':
            return True if x > Para.CROSSROAD_SIZE_LAT / 2 + 30 and Para.OFFSET_R - Para.R_OUT_0 - Para.R_OUT_1 - Para.R_OUT_2 < y < Para.OFFSET_R else False
        else:
            assert self.training_task == 'straight'
            x_trans, y_trans, _ = rotate_coordination(x, y, 0, Para.ANGLE_U - 90)
            OFFSET_U_X_trans, OFFSET_U_Y_trans, _ = rotate_coordination(Para.OFFSET_U_X, Para.OFFSET_U_Y, 0,
                                                                        Para.ANGLE_U - 90)
            return True if y_trans > OFFSET_U_Y_trans + 30 and OFFSET_U_X_trans < x_trans < OFFSET_U_X_trans + Para.U_OUT_0 + Para.U_OUT_1 else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = Para.STEER_SCALE * steer_norm
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
        if self.traffic_mode == 'auto':
            other_vector, other_mask_vector = self._construct_other_vector_short(exit_)
        else:
            other_vector, other_mask_vector = self._construct_other_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        if self.vector_noise:
            other_vector = self._add_noise_to_vector(other_vector, 'other')
            ego_vector = self._add_noise_to_vector(ego_vector, 'ego')

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

    def _add_noise_to_vector(self, vector, vec_type=None):
        """
        Enabled by the 'vector_noise' variable in this class
        Add noise to the vector of objects, whose order is (x, y, v, phi, l, w) for other and (v_x, v_y, r, x, y, phi) for ego

        Noise is i.i.d for each element in the vector, i.e. the covariance matrix is diagonal
        Different types of objs lead to different mean and var, which are defined in the 'Para' class in e2e_utils.py

        :params
            vector: np.array(6,)
            vec_type: str in ['ego', 'other']
        :return
            noise_vec: np.array(6,)
        """
        assert self.vector_noise
        assert vec_type in ['ego', 'other']
        if vec_type == 'ego':
            return vector + self.rng.multivariate_normal(Para.EGO_MEAN, Para.EGO_VAR)
        elif vec_type == 'other':
            return vector + self.rng.multivariate_normal(Para.OTHERS_MEAN, Para.OTHERS_VAR)

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
            i1_0, o1_0, i2_0, o2_0, i3_0, o3_0, i4_0, o4_0, c0, c1, c2, c3, c_w0, c_w1, c_w2, c_w3 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

            def cal_turn_rad(v):
                if not(-Para.CROSSROAD_SIZE_LAT/2 + Para.BIAS_LEFT_LAT + Para.WALK_WIDTH + Para.L/2 < v['x'] < Para.CROSSROAD_SIZE_LAT/2 - Para.L/2 - Para.WALK_WIDTH and Para.OFFSET_D_Y + Para.L/2 < v['y'] < Para.OFFSET_U_Y - Para.L/2):
                    turn_rad = 0.
                else:
                    start = v['route'][0]
                    end = v['route'][1]
                    if (start == name_setting['do'] and end == name_setting['ui']) or (start == name_setting['ro'] and end == name_setting['li'])\
                        or (start == name_setting['uo'] and end == name_setting['di']) or (start == name_setting['lo'] and end == name_setting['ri']):
                        turn_rad = 0.
                    elif (start == name_setting['do'] and end == name_setting['ri']) or (start == name_setting['ro'] and end == name_setting['ui'])\
                        or (start == name_setting['uo'] and end == name_setting['li']):
                        turn_rad = -1/17.
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        turn_rad = -1/16.
                    elif start == name_setting['do'] and end == name_setting['li']:
                        turn_rad = 1/sqrt((v['x']-(-22.7))**2 + (v['y']-(-21.4))**2)
                    elif start == name_setting['ro'] and end == name_setting['di']:
                        turn_rad = 1/sqrt((v['x']-23.)**2 + (v['y']-(-27.))**2)
                    elif start == name_setting['uo'] and end == name_setting['ri']:   # 'ur'
                        turn_rad = 1/sqrt((v['x']-24.)**2 + (v['y']-24.)**2)
                    elif start == name_setting['lo'] and end == name_setting['ui']:   # 'lu'
                        turn_rad = 1/sqrt((v['x']-(-21))**2 + (v['y']-35)**2)
                    else:
                        turn_rad = 0.
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
                    # c0 walk
                    x1_0, y1_0 = Para.OFFSET_U_X - (Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * sin(Para.ANGLE_U * pi / 180), \
                             Para.OFFSET_U_Y + (Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * cos(Para.ANGLE_U * pi / 180),
                    x2_0, y2_0 = Para.OFFSET_U_X + (Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * sin(Para.ANGLE_U * pi / 180), \
                             Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * cos(Para.ANGLE_U * pi / 180),
                    x3_0, y3_0 = x2_0 - Para.WALK_WIDTH * cos(Para.ANGLE_U * pi / 180), y2_0 - Para.WALK_WIDTH * sin(Para.ANGLE_U * pi / 180)
                    x4_0, y4_0 = x1_0 - Para.WALK_WIDTH * cos(Para.ANGLE_U * pi / 180), y1_0 - Para.WALK_WIDTH * sin(Para.ANGLE_U * pi / 180)
                    # c2 walk
                    x1_2, y1_2 = Para.OFFSET_D_X - (Para.D_OUT_0 + Para.D_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * sin(Para.ANGLE_D * pi / 180), \
                             Para.OFFSET_D_Y + (Para.D_OUT_0 + Para.D_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * cos(Para.ANGLE_D * pi / 180),
                    x2_2, y2_2 = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * sin(Para.ANGLE_D * pi / 180), \
                             Para.OFFSET_D_Y - (Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * cos(Para.ANGLE_D * pi / 180),
                    x3_2, y3_2 = x2_2 + Para.WALK_WIDTH * cos(Para.ANGLE_D * pi / 180), y2_2 + Para.WALK_WIDTH * sin(Para.ANGLE_D * pi / 180)
                    x4_2, y4_2 = x1_2 + Para.WALK_WIDTH * cos(Para.ANGLE_D * pi / 180), y1_2 + Para.WALK_WIDTH * sin(Para.ANGLE_D * pi / 180)
                    # c1 walk
                    x1_1, y1_1 = Para.CROSSROAD_SIZE_LAT / 2, \
                                 Para.OFFSET_R + Para.R_GREEN + (Para.R_IN_0 + Para.R_IN_1 + Para.R_IN_2 + Para.R_IN_3) + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH
                    x2_1, y2_1 = Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH - 14, \
                                 Para.OFFSET_R + Para.R_GREEN + (Para.R_IN_0 + Para.R_IN_1 + Para.R_IN_2 + Para.R_IN_3) + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH
                    x3_1, y3_1 = Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH - 14, \
                                 Para.OFFSET_R - 18
                    x4_1, y4_1 = Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R - 18
                    # c3 walk
                    x1_3, y1_3 = -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, \
                                 Para.OFFSET_U_Y
                    x2_3, y2_3 = -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH + 14 + Para.BIAS_LEFT_LAT, \
                                 Para.OFFSET_U_Y
                    x3_3, y3_3 = -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH + 14 + Para.BIAS_LEFT_LAT, \
                                 Para.OFFSET_L - (Para.L_IN_0 + Para.L_IN_1 + Para.L_IN_2 + Para.L_IN_3) - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH
                    x4_3, y4_3 = -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, \
                                 Para.OFFSET_L - (Para.L_IN_0 + Para.L_IN_1 + Para.L_IN_2 + Para.L_IN_3) - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH
                    if if_inPoly([(x1_0, y1_0), (x2_0, y2_0), (x3_0, y3_0), (x4_0, y4_0)], (v['x'], v['y'])):
                        c0.append(v)
                    elif if_inPoly([(x1_1, y1_1), (x2_1, y2_1), (x3_1, y3_1), (x4_1, y4_1)], (v['x'], v['y'])):
                        c1.append(v)
                    elif if_inPoly([(x1_2, y1_2), (x2_2, y2_2), (x3_2, y3_2), (x4_2, y4_2)], (v['x'], v['y'])):
                        c2.append(v)
                    elif if_inPoly([(x1_3, y1_3), (x2_3, y2_3), (x3_3, y3_3), (x4_3, y4_3)], (v['x'], v['y'])):
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
                du_b = list(filter(lambda v: ego_y - 6 < v['y'] < Para.OFFSET_R + Para.R_GREEN + Para.R_IN_0 + Para.R_IN_1 + Para.R_IN_2 + Para.R_IN_3, du_b))
            elif task == 'right':
                du_b = list(filter(lambda v: max(Para.OFFSET_R - 18, ego_y - 7) < v['y'] < min(Para.OFFSET_R, ego_y + 3) and ego_x + 20 > v['x'] > ego_x - Para.L, du_b))
            ud_b = list(filter(lambda v: max(ego_y - 3, Para.OFFSET_L + Para.L_GREEN) < v['y'] < min(Para.OFFSET_U_Y, ego_y + 7) and ego_x + Para.L > v['x'] > ego_x - 25, ud_b))  # interest of left
            lr_b = list(filter(lambda v: 0 < v['x'] < min(Para.CROSSROAD_SIZE_LAT / 2 + 40, ego_x + 10), lr_b))  # interest of right

            # fetch person in range
            c0 = list(filter(lambda v: Para.OFFSET_U_X - 4 < v['x'] and v['y'] > ego_y - Para.L, c0))  # interest of straight

            c1_du = list(filter(lambda v: (v['phi'] > 0) and max(Para.OFFSET_R - 18, ego_y - 7) < v['y'] < min(Para.OFFSET_R, ego_y + 3) and ego_x + 20 > v['x'] > ego_x - Para.L, c1))  # interest of right
            c1_ud = list(filter(lambda v: (v['phi'] < 0) and max(Para.OFFSET_R - Para.R_OUT_0 - Para.R_OUT_1 - Para.R_OUT_2, ego_y - 3) < v['y'] < min(Para.OFFSET_R + Para.R_GREEN / 2, ego_y + 7)
                                          and ego_x + 20 > v['x'] > ego_x - Para.L, c1))  # interest of right
            c1 = c1_du + c1_ud

            c2 = list(filter(lambda v: max(Para.OFFSET_D_X - 4, ego_x - 5) < v['x'] and v['y'] > ego_y - Para.L, c2))  # interest of right

            c3_du = list(filter(lambda v: (v['phi'] > 0) and max(Para.OFFSET_L + Para.L_GREEN/2, ego_y - 7) < v['y'] < min(ego_y + 3, Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2)
                                       and ego_x - 25 < v['x'] < ego_x + Para.L, c3))  # interest of left
            c3_ud = list(filter(lambda v:  (v['phi'] < 0) and max(Para.OFFSET_L + Para.L_GREEN, ego_y - 3) < v['y'] < min(ego_y + 7, Para.OFFSET_U_Y)
                                       and ego_x - 25 < v['x'] < ego_x + Para.L, c3))  # interest of left
            c3 = c3_du + c3_ud

            vir_non_veh = dict(type="bicycle_1",
                            x=7.631,
                            y=-64.6, v=0.,
                            phi=0., w=0., l=0., route=('1o', '3i'), partici_type=[0., 0., 0.], turn_rad=0., exist=False)

            tmp_non_veh = []
            for mode, num in BIKE_MODE_DICT[task].items():
                tmp_non_veh.extend(eval(mode))
            for mode, num in PERSON_MODE_DICT[task].items():
                tmp_non_veh.extend(eval(mode))
            while len(tmp_non_veh) < self.bike_num + self.person_num:
                tmp_non_veh.append(vir_non_veh)
            if len(tmp_non_veh) > self.bike_num + self.person_num:
                tmp_non_veh = sorted(tmp_non_veh, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                tmp_non_veh = tmp_non_veh[:self.bike_num + self.person_num]

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > max(ego_x - 25, -Para.CROSSROAD_SIZE_LAT / 2 - 40 + Para.BIAS_LEFT_LAT) and v['y'] > ego_y - 2, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y - 2 < v['y'] < Para.OFFSET_U_Y + 40 and v['x'] < ego_x + 5, du))  # interest of left straight

            dr = list(filter(lambda v: v['x'] < min(ego_x + 25, Para.CROSSROAD_SIZE_LAT / 2 + 40) and v['y'] > ego_y - 8, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 10 and v['y'] < Para.OFFSET_U_Y + 40, ru))  # interest of straight

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < Para.OFFSET_U_Y + 10, ur))  # interest of straight
            elif task == 'right':
                ur = list(filter(lambda v: ego_x - 20 < v['x'] < min(ego_x + 25, Para.CROSSROAD_SIZE_LAT / 2 + 40) and v['y'] < Para.OFFSET_U_Y, ur))  # interest of right
            ud = list(filter(lambda v: max(ego_y - 5, Para.OFFSET_D_Y) < v['y'] < Para.OFFSET_U_Y and ego_x + 5 > v['x'], ud))  # interest of left
            ul = list(filter(lambda v: max(-Para.CROSSROAD_SIZE_LAT / 2 - 40 + Para.BIAS_LEFT_LAT, ego_x - 25) < v['x'] < ego_x + 10 and v['y'] < Para.OFFSET_U_Y + 10, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT < v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 40, lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            vir_veh = dict(type="car_1",  x=7.631, y=-64.6, v=0., phi=0., w=0., l=0., route=('1o', '2i'),
                           partici_type=[0., 0., 0.], turn_rad=0., exist=False)

            tmp_v = []
            for mode, num in VEHICLE_MODE_DICT[task].items():
                tmp_v.extend(eval(mode))
            while len(tmp_v) < self.veh_num:
                tmp_v.append(vir_veh)
            if len(tmp_v) > self.veh_num:
                tmp_v = sorted(tmp_v, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2), -v['x']))
                tmp_v = tmp_v[:self.veh_num]

            tmp = tmp_non_veh + tmp_v
            return tmp

        self.interested_other = filter_interested_other(self.all_other, self.training_task)

        for other in self.interested_other:
            other_x, other_y, other_v, other_phi, other_l, other_w, other_type, other_turn_rad, other_mask = \
                other['x'], other['y'], other['v'], other['phi'], other['l'], other['w'], other['partici_type'], other[
                    'turn_rad'], other['exist']
            other_vector.extend(
                [other_x, other_y, other_v, other_phi, other_l, other_w] + other_type + [other_turn_rad])
            other_mask_vector.append(other_mask)

        return np.array(other_vector, dtype=np.float32), np.array(other_mask_vector, dtype=np.float32)

    def _construct_other_vector_hand_traffic(self, exit_='D'):
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
            ped_all, bike_all, veh_all = [], [], []

            def cal_turn_rad(v):
                if not(-Para.CROSSROAD_SIZE_LAT/2 + Para.L/2 < v['x'] < Para.CROSSROAD_SIZE_LAT/2 - Para.L/2 and Para.OFFSET_D_Y + Para.L/2 < v['y'] < Para.OFFSET_U_Y - Para.L/2):
                    turn_rad = 0.
                else:
                    start = v['route'][0]
                    end = v['route'][1]
                    if (start == name_setting['do'] and end == name_setting['ui']) or (start == name_setting['ro'] and end == name_setting['li'])\
                        or (start == name_setting['uo'] and end == name_setting['di']) or (start == name_setting['lo'] and end == name_setting['ri']):
                        turn_rad = 0.
                    elif (start == name_setting['do'] and end == name_setting['ri']) or (start == name_setting['ro'] and end == name_setting['ui'])\
                        or (start == name_setting['uo'] and end == name_setting['li']):
                        turn_rad = -1/17.
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        turn_rad = -1/16.
                    elif start == name_setting['do'] and end == name_setting['li']:
                        turn_rad = 1/sqrt((v['x']-(-22.7))**2 + (v['y']-(-21.4))**2)
                    elif start == name_setting['ro'] and end == name_setting['di']:
                        turn_rad = 1/sqrt((v['x']-23.)**2 + (v['y']-(-27.))**2)
                    elif start == name_setting['uo'] and end == name_setting['ri']:   # 'ur'
                        turn_rad = 1/sqrt((v['x']-24.)**2 + (v['y']-24.)**2)
                    elif start == name_setting['lo'] and end == name_setting['ui']:   # 'lu'
                        turn_rad = 1/sqrt((v['x']-(-21))**2 + (v['y']-35)**2)
                    else:
                        turn_rad = 0.
                return turn_rad

            for v in vs:
                if (-Para.CROSSROAD_SIZE_LAT / 2 - 30 < v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 30 and
                        Para.OFFSET_D_Y - 30 < v['y'] < Para.OFFSET_U_Y + 30):
                    if v['type'] in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                        v.update(partici_type=[1., 0., 0.], turn_rad=0.0, exist=True)
                        bike_all.append(v)

                    elif v['type'] == 'DEFAULT_PEDTYPE':
                        v.update(partici_type=[0., 1., 0.], turn_rad=0.0, exist=True)
                        ped_all.append(v)
                    else:
                        v.update(partici_type=[0., 0., 1.], turn_rad=cal_turn_rad(v), exist=True)
                        veh_all.append(v)

            mode2fillvalue_b = dict(type="bicycle_1", x=Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH / 2) * sin(Para.ANGLE_D * pi / 180),
                          y=Para.OFFSET_D_Y - 35, v=0., phi=0., w=0., l=0., route=('1o', '3i'), partici_type=[0., 0., 0.], turn_rad=0., exist=False)

            mode2fillvalue_p = dict(type='DEFAULT_PEDTYPE',
                        x=Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH / 2) * sin(Para.ANGLE_D * pi / 180),
                        y=Para.OFFSET_D_Y - 35, v=0, phi=0., w=0., l=0., road="0_c1", partici_type=[0., 0., 0.], turn_rad=0., exist=False)

            mode2fillvalue_v = dict(type="car_1", x=Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH / 2) * sin(Para.ANGLE_D * pi / 180),
                        y=Para.OFFSET_D_Y - 35, v=0, phi=0., w=0., l=0., route=('1o', '4i'), partici_type=[0., 0., 0.],
                        turn_rad=0., exist=False)

            while len(bike_all) < self.bike_num:
                bike_all.append(mode2fillvalue_b)
            if len(bike_all) > self.bike_num:
                bike_all_sorted = sorted(bike_all, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                bike_all = bike_all_sorted[:self.bike_num]

            while len(ped_all) < self.person_num:
                ped_all.append(mode2fillvalue_p)
            if len(ped_all) > self.person_num:
                ped_all_sorted = sorted(ped_all, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                ped_all = ped_all_sorted[:self.person_num]

            while len(veh_all) < self.veh_num:
                veh_all.append(mode2fillvalue_v)
            if len(veh_all) > self.veh_num:
                veh_all_sorted = sorted(veh_all, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2), -v['x']))
                veh_all = veh_all_sorted[:self.veh_num]

            tmp = bike_all + ped_all + veh_all
            return tmp

        self.interested_other = filter_interested_other(self.all_other, self.training_task)

        for other in self.interested_other:
            other_x, other_y, other_v, other_phi, other_l, other_w, other_type, other_turn_rad, other_mask = \
                other['x'], other['y'], other['v'], other['phi'], other['l'], other['w'], other['partici_type'], other[
                    'turn_rad'], other['exist']
            other_vector.extend(
                [other_x, other_y, other_v, other_phi, other_l, other_w] + other_type + [other_turn_rad])
            other_mask_vector.append(other_mask)

        return np.array(other_vector, dtype=np.float32), np.array(other_mask_vector, dtype=np.float32)

    def _reset_init_state(self, light_phase):
        if self.training_task == 'left':
            if light_phase == 'green':
                random_index = int(np.random.random() * 2750) + 450
            else:
                random_index = int(np.random.random() * 200) + 700
        elif self.training_task == 'straight':
            if light_phase == 'green':
                random_index = int(np.random.random() * (1200 + 1200)) + 500
            else:
                random_index = int(np.random.random() * 200) + 700
        else:
            random_index = int(np.random.random() * (800 + 1000)) + 500

        if self.mode == 'testing' and self.traffic_mode == 'user':
            random_index = MODE2INDEX_TEST[self.traffic_case] + int(np.random.random() * 100)
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
        render(light_phase=self.light_phase, all_other=self.all_other, interested_other=self.interested_other,
               attn_weights=attn_weights, obs=self.obs, ref_path=self.ref_path,
               future_n_point=self.future_n_point, action=self.action, done_type=self.done_type,
               reward_info=self.reward_dict, hist_posi=None, path_values=None)
        plt.show()
        plt.pause(0.001)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    env = IdcRealEnv()
    env_model = IdcRealModel()
    obs, _ = env.reset()
    i = 0
    while i < 100000:
        for j in range(200):
            i += 1
            action = np.array([0.3, 0.6 + np.random.rand(1)*0.8], dtype=np.float32) # np.random.rand(1)*0.1 - 0.05
            obs, reward, done, info = env.step(action)
            env_model.reset(obs[np.newaxis, :], info['future_n_point'][np.newaxis, :])
            for _ in range(10):
                obs_model, reward_dict = env_model.rollout(action[np.newaxis, :])
                env_model.render()
            env.render()
            if done:
                print(env.done_type)
                break
        obs, _ = env.reset()
        env.render()


if __name__ == '__main__':
    test_end2end()
