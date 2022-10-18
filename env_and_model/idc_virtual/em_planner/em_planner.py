#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/3/26
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: em_planner.py.py
# =====================================
import time
from ctypes import *

import numpy as np
import ray

from env_and_model.idc_virtual.em_planner.endtoend import IdcVirtualEnv
from env_and_model.idc_virtual.endtoend_env_utils import *
from env_and_model.idc_virtual.hierarchical_decision.multi_path_generator import MultiPathGenerator
from env_and_model.idc_virtual.utils.bezier_curve import CubicBezierCurve


class EgoState(Structure):
    _fields_ = [("vx", c_double),
                ("r", c_double),
                ("x", c_double),
                ("y", c_double),
                ("heading", c_double),
                ("acc", c_double),
                ("t", c_double)]


class OtherState(Structure):
    _fields_ = [("id", c_int),
                ("x", c_double),
                ("y", c_double),
                ("v", c_double),
                ("heading", c_double),
                ("l", c_double),
                ("w", c_double),
                ("cor_xs", (c_double * 4)),
                ("cor_ys", (c_double * 4)),
                ("type", c_int),
                ("turn_rad", c_double),
                ("t", c_double),
                ("period", c_double)
                ]


class AllObst(Structure):
    _fields_ = [("other_states", (OtherState * 300)),
                ("valid_num", c_int),
                ]


ref_point_num = 720


class ReferLine(Structure):
    _fields_ = [("x", (c_double * ref_point_num)),
                ("y", (c_double * ref_point_num)),
                ("heading", (c_double * ref_point_num)),
                ("kappa", (c_double * ref_point_num)),
                ("dkappa", (c_double * ref_point_num)),
                ]


class ReferLineCost(Structure):
    _fields_ = [("x1", c_double),
                ("x2", c_double),
                ("x3", c_double),
                ]


class SpeedLimits(Structure):
    _fields_ = [("start_s", (c_double * 3)),
                ("end_s", (c_double * 3)),
                ("limit", (c_double * 3)),
                ]


traj_point_num = 200


class Traj(Structure):
    _fields_ = [("valid_num", c_int),
                ("x", (c_double * traj_point_num)),
                ("y", (c_double * traj_point_num)),
                ("theta", (c_double * traj_point_num)),
                ("kappa", (c_double * traj_point_num)),
                ("s", (c_double * traj_point_num)),
                ("dkappa", (c_double * traj_point_num)),
                ("v", (c_double * traj_point_num)),
                ("a", (c_double * traj_point_num)),
                ("relative_time", (c_double * traj_point_num)),
                ]


class ControlCmd(Structure):
    _fields_ = [("steering_rate", c_double),
                ("steering_target", c_double),
                ("acceleration", c_double),
                ]


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class EmPlanner:
    def __init__(self):
        self.em_planner_so = cdll.LoadLibrary(CURRENT_DIR + '/planning.so')
        self.em_planner_so.Init()
        self.t = 0.
        self.dt = 0.1
        self.control_step = 0
        self.debug_things = {}
        self.prev_act = [0., 0.]
        self.env = IdcVirtualEnv()
        self.obs = None
        self.all_info = None
        self.task = None
        self.light = None

    def runonce(self, obs, interested_others_for_em_planner):
        self._construct_vehicle_state(obs, self.prev_act)
        self._construct_predict(interested_others_for_em_planner)
        start_t = time.time()
        self.em_planner_so.RunOnce(c_double(self.t))
        planning_time_ms = (time.time() - start_t)*1000.
        print('em planner time(ms): ', planning_time_ms)
        self.em_planner_so.PassTrajectoryInControl()
        self._get_debug_things()

        self._construct_vehicle_state(obs, self.prev_act, 'control')
        control_cmd = ControlCmd()
        start_t = time.time()
        self.em_planner_so.ProduceControlCommand(c_double(self.t), byref(control_cmd))
        control_time_ms = (time.time() - start_t) * 1000.
        print('control time(ms): ', control_time_ms)
        self.t += self.dt
        self.control_step += 1
        comp_info = {'cal_time_ms': planning_time_ms+control_time_ms,
                     'planning_time_ms': planning_time_ms,
                     'control_time_ms': control_time_ms,
                     'a_x': control_cmd.acceleration,
                     'pass_time_s': 0.1}
        return self.action_trans(control_cmd), comp_info

    def action_trans(self, control_cmd):
        steering_target = control_cmd.steering_target / 100. * 0.5  # todo common/data/mkz_config.pb
        acceleration = control_cmd.acceleration
        return np.array([steering_target, acceleration], np.float32)

    def _get_ref_lines(self, task, light):
        ref_line_list = []
        speed_limits_list = []
        l_front, _, l_back = 40, 40, 100
        if light == 'green':
            speed_limit_front, speed_limit_mid, speed_limit_back = 6.66, 4.16, 6.66  # todo: when it is higher, the em planner may cannot solve the tasks
        else:
            speed_limit_front, speed_limit_mid, speed_limit_back = 6.66, 0., 0.
        num_front, num_mid, num_back = 160, 160, 400
        assert num_front+num_mid+num_back == ref_point_num
        if task == 'left':
            end_offsets = [Para.LANE_WIDTH * (i + 0.5) for i in range(1)]
            start_offsets = [Para.LANE_WIDTH * 0.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE / 2
                    control_point4 = -Para.CROSSROAD_SIZE / 2, end_offset
                    control_point2, control_point3 = \
                        get_bezier_control_points(control_point1[0], control_point1[1], 0.5 * np.pi,
                                                  control_point4[0], control_point4[1], np.pi)
                    mid = CubicBezierCurve(control_point1, control_point2, control_point3, control_point4)
                    ts = np.linspace(0., 1.0, num_mid)
                    mid_x, mid_y, mid_phi, mid_kappa, mid_dkappa, l_mid =\
                        mid.x(ts), mid.y(ts), mid.phi(ts), mid.kappa(ts), mid.dkappa(ts), mid.length(1.0)

                    start_x = Para.LANE_WIDTH / 2 * np.ones(shape=(num_front+1,), dtype=np.float32)[:-1]
                    start_y = np.linspace(-Para.CROSSROAD_SIZE / 2 - l_front, -Para.CROSSROAD_SIZE / 2, num_front+1, dtype=np.float32)[:-1]
                    start_phi, start_kappa, start_dkappa = np.pi/2*np.ones_like(start_x), np.zeros_like(start_x), np.zeros_like(start_x)
                    end_x = np.linspace(-Para.CROSSROAD_SIZE / 2, -Para.CROSSROAD_SIZE / 2 - l_back, num_back+1, dtype=np.float32)[1:]
                    end_y = end_offset * np.ones(shape=(num_back+1,), dtype=np.float32)[1:]
                    end_phi, end_kappa, end_dkappa = np.pi*np.ones_like(end_x), np.zeros_like(end_x), np.zeros_like(end_x)

                    xs = np.concatenate([start_x, mid_x, end_x], 0)
                    ys = np.concatenate([start_y, mid_y, end_y], 0)
                    phis = np.concatenate([start_phi, mid_phi, end_phi], 0)
                    kappas = np.concatenate([start_kappa, mid_kappa, end_kappa], 0)
                    dkappas = np.concatenate([start_dkappa, mid_dkappa, end_dkappa], 0)
                    ref_line, speed_limits = ReferLine(), SpeedLimits()
                    for i in range(ref_point_num):
                        ref_line.x[i] = xs[i]
                        ref_line.y[i] = ys[i]
                        ref_line.heading[i] = phis[i]
                        ref_line.kappa[i] = kappas[i]
                        ref_line.dkappa[i] = dkappas[i]
                    ref_line_list.append(ref_line)
                    tmp = [[0., l_front, speed_limit_front],
                           [l_front, l_front + l_mid, speed_limit_mid],
                           [l_front + l_mid, l_front + l_mid + l_back, speed_limit_back]]
                    for i, speed_limit in enumerate(tmp):
                        speed_limits.start_s[i] = speed_limit[0]
                        speed_limits.end_s[i] = speed_limit[1]
                        speed_limits.limit[i] = speed_limit[2]
                    speed_limits_list.append(speed_limits)

        elif task == 'straight':
            end_offsets = [Para.LANE_WIDTH * (i + 0.5) for i in range(1, 2)]
            start_offsets = [Para.LANE_WIDTH * 1.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE / 2
                    control_point4 = end_offset, Para.CROSSROAD_SIZE / 2
                    control_point2, control_point3 = \
                        get_bezier_control_points(control_point1[0], control_point1[1], 0.5 * np.pi,
                                                  control_point4[0], control_point4[1], 0.5 * np.pi)
                    mid = CubicBezierCurve(control_point1, control_point2, control_point3, control_point4)
                    ts = np.linspace(0., 1.0, num_mid)
                    mid_x, mid_y, mid_phi, mid_kappa, mid_dkappa, l_mid = \
                        mid.x(ts), mid.y(ts), mid.phi(ts), mid.kappa(ts), mid.dkappa(ts), mid.length(1.)

                    start_x = start_offset * np.ones(shape=(num_front+1,), dtype=np.float32)[:-1]
                    start_y = np.linspace(-Para.CROSSROAD_SIZE / 2 - l_front, -Para.CROSSROAD_SIZE / 2, num_front+1, dtype=np.float32)[:-1]
                    start_phi, start_kappa, start_dkappa = np.pi/2*np.ones_like(start_x), np.zeros_like(start_x), np.zeros_like(start_x)
                    end_x = end_offset * np.ones(shape=(num_back+1,), dtype=np.float32)[1:]
                    end_y = np.linspace(Para.CROSSROAD_SIZE / 2, Para.CROSSROAD_SIZE / 2 + l_back, num_back+1, dtype=np.float32)[1:]
                    end_phi, end_kappa, end_dkappa = np.pi/2*np.ones_like(end_x), np.zeros_like(end_x), np.zeros_like(end_x)

                    xs = np.concatenate([start_x, mid_x, end_x], 0)
                    ys = np.concatenate([start_y, mid_y, end_y], 0)
                    phis = np.concatenate([start_phi, mid_phi, end_phi], 0)
                    kappas = np.concatenate([start_kappa, mid_kappa, end_kappa], 0)
                    dkappas = np.concatenate([start_dkappa, mid_dkappa, end_dkappa], 0)
                    ref_line, speed_limits = ReferLine(), SpeedLimits()
                    for i in range(ref_point_num):
                        ref_line.x[i] = xs[i]
                        ref_line.y[i] = ys[i]
                        ref_line.heading[i] = phis[i]
                        ref_line.kappa[i] = kappas[i]
                        ref_line.dkappa[i] = dkappas[i]
                    ref_line_list.append(ref_line)
                    tmp = [[0., l_front, speed_limit_front],
                           [l_front, l_front + l_mid, speed_limit_mid],
                           [l_front + l_mid, l_front + l_mid + l_back, speed_limit_back]]
                    for i, speed_limit in enumerate(tmp):
                        speed_limits.start_s[i] = speed_limit[0]
                        speed_limits.end_s[i] = speed_limit[1]
                        speed_limits.limit[i] = speed_limit[2]
                    speed_limits_list.append(speed_limits)
        else:
            # end_offsets = [-Para.LANE_WIDTH * 2.5, -Para.LANE_WIDTH * 1.5,]
            end_offsets = [-Para.LANE_WIDTH * 2.5]
            start_offsets = [Para.LANE_WIDTH * 2.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE / 2
                    control_point4 = Para.CROSSROAD_SIZE / 2, end_offset
                    control_point2, control_point3 = \
                        get_bezier_control_points(control_point1[0], control_point1[1], 0.5 * np.pi,
                                                  control_point4[0], control_point4[1], 0.)
                    mid = CubicBezierCurve(control_point1, control_point2, control_point3, control_point4)
                    ts = np.linspace(0., 1.0, num_mid)
                    mid_x, mid_y, mid_phi, mid_kappa, mid_dkappa, l_mid = \
                        mid.x(ts), mid.y(ts), mid.phi(ts), mid.kappa(ts), mid.dkappa(ts), mid.length(1.)

                    start_x = start_offset * np.ones(shape=(num_front+1,), dtype=np.float32)[:-1]
                    start_y = np.linspace(-Para.CROSSROAD_SIZE / 2 - l_front, -Para.CROSSROAD_SIZE / 2, num_front+1, dtype=np.float32)[:-1]
                    start_phi, start_kappa, start_dkappa = np.pi/2*np.ones_like(start_x), np.zeros_like(start_x), np.zeros_like(start_x)
                    end_x = np.linspace(Para.CROSSROAD_SIZE / 2, Para.CROSSROAD_SIZE / 2 + l_back, num_back+1, dtype=np.float32)[1:]
                    end_y = end_offset * np.ones(shape=(num_back+1,), dtype=np.float32)[1:]
                    end_phi, end_kappa, end_dkappa = 0.*np.ones_like(end_x), np.zeros_like(end_x), np.zeros_like(end_x)

                    xs = np.concatenate([start_x, mid_x, end_x], 0)
                    ys = np.concatenate([start_y, mid_y, end_y], 0)
                    phis = np.concatenate([start_phi, mid_phi, end_phi], 0)
                    kappas = np.concatenate([start_kappa, mid_kappa, end_kappa], 0)
                    dkappas = np.concatenate([start_dkappa, mid_dkappa, end_dkappa], 0)
                    ref_line, speed_limits = ReferLine(), SpeedLimits()
                    for i in range(ref_point_num):
                        ref_line.x[i] = xs[i]
                        ref_line.y[i] = ys[i]
                        ref_line.heading[i] = phis[i]
                        ref_line.kappa[i] = kappas[i]
                        ref_line.dkappa[i] = dkappas[i]
                    ref_line_list.append(ref_line)
                    tmp = [[0., l_front, speed_limit_front],
                           [l_front, l_front + l_mid, speed_limit_mid],
                           [l_front + l_mid, l_front + l_mid + l_back, speed_limit_back]]
                    for i, speed_limit in enumerate(tmp):
                        speed_limits.start_s[i] = speed_limit[0]
                        speed_limits.end_s[i] = speed_limit[1]
                        speed_limits.limit[i] = speed_limit[2]
                    speed_limits_list.append(speed_limits)

        return ref_line_list, speed_limits_list

    def _construct_vehicle_state(self, obs, act, planning_or_control='planning'):
        v_x, v_y, r, x, y, phi = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
        _, acc = act[0], act[1]
        ego_state = EgoState()
        ego_state.vx = v_x
        ego_state.r = r
        ego_state.x = x
        ego_state.y = y
        ego_state.heading = phi * np.pi / 180
        ego_state.acc = acc
        ego_state.t = self.t
        if planning_or_control == 'planning':
            self.em_planner_so.ConstructVehicleState(byref(ego_state))
        else:
            self.em_planner_so.ConstructVehicleStateForControl(byref(ego_state))

    def _construct_predict(self, interested_others_for_em_planner):
        def cal_corner_points(l, w, x, y, phi):
            x0, y0, _ = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, _ = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, _ = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, _ = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return [x0, x1, x2, x3], [y0, y1, y2, y3]

        self.em_planner_so.ClearPrediction()
        for i, other in enumerate(interested_others_for_em_planner):
            other_x, other_y, other_v, other_phi, other_l, other_w, other_type, other_turn_rad = other
            other_state = OtherState()
            other_state.id = i
            other_state.x = other_x
            other_state.y = other_y
            other_state.v = other_v
            other_state.heading = other_phi * np.pi / 180
            other_state.l = other_l
            other_state.w = other_w
            cor_xs, cor_ys = cal_corner_points(other_l, other_w, other_x, other_y, other_phi)
            other_state.cor_xs[0], other_state.cor_xs[1], other_state.cor_xs[2], other_state.cor_xs[3]\
                = cor_xs[0], cor_xs[1], cor_xs[2], cor_xs[3]
            other_state.cor_ys[0], other_state.cor_ys[1], other_state.cor_ys[2], other_state.cor_ys[3] \
                = cor_ys[0], cor_ys[1], cor_ys[2], cor_ys[3]
            other_state.type = other_type
            other_state.turn_rad = other_turn_rad
            other_state.t = self.t
            other_state.period = 5.
            self.em_planner_so.AddPredictionObstacle(byref(other_state))

    def _construct_reference_lines(self, task, light):
        self.em_planner_so.ClearReferenceLines()
        ref_line_list, speed_limits_list = self._get_ref_lines(task, light)
        for ref_line, speed_limits in zip(ref_line_list, speed_limits_list):
            self.em_planner_so.AddReferenceLine(byref(ref_line), byref(speed_limits))

    def _get_debug_things(self, ):
        traj = Traj()
        drive_refer_line = ReferLine()
        referline_cost = ReferLineCost()
        all_obst = AllObst()
        self.em_planner_so.GetDebugThings(byref(traj),
                                          byref(drive_refer_line),
                                          byref(referline_cost),
                                          byref(all_obst))
        traj = {'x': np.array([traj.x[i] for i in range(traj.valid_num)]),
                'y': np.array([traj.y[i] for i in range(traj.valid_num)]),
                'theta': np.array([traj.theta[i] for i in range(traj.valid_num)]),
                'kappa': np.array([traj.kappa[i] for i in range(traj.valid_num)]),
                's': np.array([traj.s[i] for i in range(traj.valid_num)]),
                'dkappa': np.array([traj.dkappa[i] for i in range(traj.valid_num)]),
                'v': np.array([traj.v[i] for i in range(traj.valid_num)]),
                'a': np.array([traj.a[i] for i in range(traj.valid_num)]),
                'relative_time': np.array([traj.relative_time[i] for i in range(traj.valid_num)]),
                }
        drive_refer_line = {'x': np.array([drive_refer_line.x[i] for i in range(ref_point_num)]),
                            'y': np.array([drive_refer_line.y[i] for i in range(ref_point_num)]),
                            }
        referline_cost = np.array([referline_cost.x1, referline_cost.x2, referline_cost.x3])
        all_obst = {'x': np.array([all_obst.other_states[i].x for i in range(all_obst.valid_num)]),
                    'y': np.array([all_obst.other_states[i].y for i in range(all_obst.valid_num)]),}
        self.debug_things = {'traj': traj,
                             'drive_refer_line': drive_refer_line,
                             'referline_cost': referline_cost,
                             'all_obst': all_obst}

    def reset(self):
        self.obs, self.all_info = self.env.reset()
        self.task = self.env.training_task
        self.light = LIGHT_PHASE_TO_GREEN_OR_RED[self.all_info['light_phase']]
        self._construct_reference_lines(self.task, self.light)

    def run_an_episode(self, is_render=False):
        self.reset()
        done = False
        episode_info = []
        while not done:
            interested_others_for_em_planner = self.all_info['interested_others_for_em_planner']
            interested_others_for_em_planner = sorted(interested_others_for_em_planner,
                                                      key=lambda x: np.square(x[0]-self.obs[3])+np.square(x[1]-self.obs[4]))
            action, comp_info = self.runonce(self.obs, interested_others_for_em_planner[:50])  # todo the more the value the more time will be taken
            debug_things = self.debug_things
            best_idx = int(np.argmin(debug_things['referline_cost']))
            stg = MultiPathGenerator()
            path_list = stg.generate_path(self.task, self.light)
            self.env.set_traj(path_list[best_idx])
            self.obs, rew, done, self.all_info = self.env.step(action)
            comp_info['done'] = done
            comp_info['done_type'] = self.env.done_type
            episode_info.append(comp_info)
            if is_render:
                self.env.render(traj=debug_things['traj'],
                                ref_path=None,  # debug_things['drive_refer_line'],
                                referline_cost=debug_things['referline_cost'],
                                all_obst=debug_things['all_obst'])
        return episode_info


def main():
    em_planner = EmPlanner()
    episode_info = em_planner.run_an_episode(is_render=True)
    print(episode_info)


if __name__ == "__main__":
    main()


