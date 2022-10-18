#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: traffic.py
# =====================================

import copy
import os
import random
import sys
from collections import defaultdict
from math import fabs, cos, sin, pi

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
from traci.exceptions import FatalTraCIError
from env_and_model.idc_virtual_veh.endtoend_env_utils import *

SUMO_BINARY = checkBinary('sumo')
SIM_PERIOD = 1.0 / 10


class Traffic(object):
    def __init__(self, step_length, mode, init_n_ego_dict):  # mode 'display' or 'training'
        self.random_traffic = None
        self.sim_time = 0
        self.n_ego_others = defaultdict(list)
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.light_phase = None
        self.n_ego_dict = init_n_ego_dict
        self.training_light_phase = None
        self.mode = mode
        # training 意味着不变灯 但reset时可以是红灯或绿灯

        try:
            traci.start(
                [SUMO_BINARY, "-c", SUMOCFG_DIR,
                 "--step-length", self.step_time_str,
                 # "--lateral-resolution", "3.5",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', '23423' if seed is None else str(int(seed))
                 ], numRetries=5)  # '--seed', str(int(seed))
        except FatalTraCIError:
            print('Retry by other port')
            port = sumolib.miscutils.getFreeSocketPort()
            traci.start(
                [SUMO_BINARY, "-c", SUMOCFG_DIR,
                 "--step-length", self.step_time_str,
                 "--lateral-resolution", "3.5",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', '23423' if seed is None else str(int(seed))
                 ], port=port, numRetries=5)  # '--seed', str(int(seed))

        traci.junction.subscribeContext(objectID='0', domain=traci.constants.CMD_GET_VEHICLE_VARIABLE, dist=10000.0,
                                        varIDs=[traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_SPEED_LAT,
                                                traci.constants.VAR_TYPE,
                                                # traci.constants.VAR_EMERGENCY_DECEL,
                                                # traci.constants.VAR_LANE_INDEX,
                                                # traci.constants.VAR_LANEPOSITION,
                                                # traci.constants.VAR_EDGES,
                                                # traci.constants.VAR_ROAD_ID,
                                                traci.constants.VAR_EDGES,
                                                # traci.constants.VAR_NEXT_EDGE,
                                                # traci.constants.VAR_ROUTE_INDEX
                                                ], begin=0.0, end=2147483647.0)

        traci.junction.subscribeContext(objectID='0', domain=traci.constants.CMD_GET_PERSON_VARIABLE, dist=10000.0,
                                        varIDs=[traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                # traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_TYPE,
                                                # traci.constants.VAR_EMERGENCY_DECEL,
                                                # traci.constants.VAR_LANE_INDEX,
                                                # traci.constants.VAR_LANEPOSITION,
                                                # traci.constants.VAR_EDGES,
                                                traci.constants.VAR_ROAD_ID,
                                                # traci.constants.VAR_NEXT_EDGE,
                                                # traci.constants.VAR_ROUTE_ID,
                                                # traci.constants.VAR_ROUTE_INDEX
                                                ], begin=0.0, end=2147483647.0)

        self.init_step()
        self.prev_training_light_phase = 0

    def init_step(self):
        # TODO(guanyang): determine the 250
        while traci.simulation.getTime() < 250:
            if traci.simulation.getTime() < 249:
                traci.trafficlight.setPhase('0', 2)
            else:
                traci.trafficlight.setPhase('0', 0)
            traci.simulationStep()

    def __del__(self):
        traci.close()

    def close(self):
        traci.close()

    def add_self_car(self, n_ego_dict, with_delete=True):
        for egoID, ego_dict in n_ego_dict.items():
            ego_v_x = ego_dict['v_x']
            ego_v_y = ego_dict['v_y']
            ego_l = ego_dict['l']
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            ego_phi = ego_dict['phi']
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            edgeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            if with_delete:
                try:
                    traci.vehicle.remove(egoID)
                except traci.exceptions.TraCIException:
                    print('Don\'t worry, it\'s been handled well')
                traci.simulationStep()
                traci.vehicle.addLegacy(vehID=egoID, routeID=ego_dict['routeID'],
                                        # depart=0, pos=20, lane=lane, speed=ego_dict['v_x'],
                                        typeID='self_car')
            traci.vehicle.moveToXY(egoID, edgeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keepRoute=1)
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x ** 2 + ego_v_y ** 2))

    def generate_random_traffic(self, init_n_ego_dict):
        random_traffic = traci.junction.getContextSubscriptionResults('0')
        random_traffic = copy.deepcopy(random_traffic)
        if self.prev_training_light_phase != self.training_light_phase:
            for other in random_traffic:
                x_in_sumo, y_in_sumo = random_traffic[other][traci.constants.VAR_POSITION]
                a_in_sumo = random_traffic[other][traci.constants.VAR_ANGLE]
                other_l = random_traffic[other][traci.constants.VAR_LENGTH]
                x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, other_l)
                other_type = random_traffic[other][traci.constants.VAR_TYPE]
                if -Para.CROSSROAD_SIZE/2 < x < Para.CROSSROAD_SIZE/2 and \
                        -Para.CROSSROAD_SIZE/2 < y < Para.CROSSROAD_SIZE/2:
                    if other_type == 'DEFAULT_PEDTYPE':
                        traci.person.removeStages(other)
                    else:
                        traci.vehicle.remove(other)
            start_time = traci.simulation.getTime()
            while traci.simulation.getTime() - start_time < 50:
                traci.trafficlight.setPhase('0', self.training_light_phase)
                traci.simulationStep()
            self.prev_training_light_phase = self.training_light_phase

        self.add_self_car(init_n_ego_dict)
        random_traffic = traci.junction.getContextSubscriptionResults('0')
        random_traffic = copy.deepcopy(random_traffic)
        for ego_id in list(self.n_ego_dict.keys()):
            if ego_id in random_traffic:
                del random_traffic[ego_id]
        return random_traffic

    def init_light(self, green_prob):
        if random.random() > green_prob:
            self.training_light_phase = 3
        else:
            self.training_light_phase = 0
        traci.trafficlight.setPhase('0', self.training_light_phase)
        # traci.trafficlight.setPhaseDuration('0', 10000)
        traci.simulationStep()
        self._get_traffic_light()
        return self.light_phase

    def init_traffic(self, init_n_ego_dict):
        self.sim_time = 0
        self.n_ego_others = defaultdict(list)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.n_ego_dict = init_n_ego_dict
        traci.simulationStep()
        random_traffic = self.generate_random_traffic(init_n_ego_dict)
        self.add_self_car(init_n_ego_dict, with_delete=False)
        # 跑完以后加自车 加完自车生成交通流（然后就不能有simulationstep了）,再用with_delete=false移动自车

        # move ego to the given position and remove conflict cars
        for egoID, ego_dict in self.n_ego_dict.items():
            ego_x, ego_y, ego_v_x, ego_v_y, ego_phi, ego_l, ego_w = ego_dict['x'], ego_dict['y'], ego_dict['v_x'], \
                                                                    ego_dict['v_y'], ego_dict['phi'], ego_dict['l'], \
                                                                    ego_dict['w']
            for other in random_traffic:
                x_in_sumo, y_in_sumo = random_traffic[other][traci.constants.VAR_POSITION]
                a_in_sumo = random_traffic[other][traci.constants.VAR_ANGLE]
                other_l = random_traffic[other][traci.constants.VAR_LENGTH]
                other_v = random_traffic[other][traci.constants.VAR_SPEED]
                other_type = random_traffic[other][traci.constants.VAR_TYPE]
                # veh_sig = random_traffic[veh][traci.constants.VAR_SIGNALS]
                # 10: left and brake 9: right and brake 1: right 8: brake 0: no signal 2: left

                x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, other_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x,
                                                                                               ego_y, ego_phi)
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = \
                    shift_and_rotate_coordination(0, 0, 0,
                                                  x_in_ego_coord,
                                                  y_in_ego_coord,
                                                  a_in_ego_coord)
                if (-5 < x_in_ego_coord < 1 * (ego_v_x) + ego_l / 2. + other_l / 2. + 2 and abs(y_in_ego_coord) < 3) or \
                        (-5 < ego_x_in_veh_coord < 1 * (other_v) + ego_l / 2. + other_l / 2. + 2 and abs(
                            ego_y_in_veh_coord) < 3):
                    if other_type == 'DEFAULT_PEDTYPE':
                        traci.person.removeStages(other)
                    else:
                        traci.vehicle.remove(other)
                    # traci.vehicle.remove(vehID=veh)
                # if 0<x_in_sumo<3.5 and -22<y_in_sumo<-15:# and veh_sig!=1 and veh_sig!=9:
                #     traci.vehicle.moveToXY(veh, '4o', 1, -80, 1.85, 180,2)
                #     traci.vehicle.remove(vehID=veh)

    def _get_others(self):
        self.n_ego_others = defaultdict(list)
        other_infos = traci.junction.getContextSubscriptionResults('0')
        for egoID in self.n_ego_dict.keys():
            other_info_dict = copy.deepcopy(other_infos)
            for i, other in enumerate(other_info_dict):
                if other != egoID:
                    length = other_info_dict[other][traci.constants.VAR_LENGTH]
                    width = other_info_dict[other][traci.constants.VAR_WIDTH]
                    type = other_info_dict[other][traci.constants.VAR_TYPE]
                    if type == 'DEFAULT_PEDTYPE':
                        # TODO: 0为暂时赋值
                        route = '0 0'
                        road = other_info_dict[other][traci.constants.VAR_ROAD_ID]
                    else:
                        route = other_info_dict[other][traci.constants.VAR_EDGES]
                        road = '0'
                    if route[0] == '4i':
                        continue
                    x_in_sumo, y_in_sumo = other_info_dict[other][traci.constants.VAR_POSITION]
                    a_in_sumo = other_info_dict[other][traci.constants.VAR_ANGLE]
                    # transfer x,y,a in car coord
                    x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                    v = other_info_dict[other][traci.constants.VAR_SPEED]
                    self.n_ego_others[egoID].append(dict(type=type, x=x, y=y, v=v, phi=a, l=length,
                                                         w=width, route=route, road=road))

    def _get_traffic_light(self):
        self.light_phase = traci.trafficlight.getPhase('0')

    def sim_step(self):
        self.sim_time += SIM_PERIOD
        if self.mode == 'training':
            traci.trafficlight.setPhase('0', self.training_light_phase)
        traci.simulationStep()
        self._get_others()
        self._get_traffic_light()
        self.collision_check()
        for egoID, collision_flag in self.n_ego_collision_flag.items():
            if collision_flag:
                self.collision_flag = True
                self.collision_ego_id = egoID

    def set_own_car(self, n_ego_dict_):
        assert len(self.n_ego_dict) == len(n_ego_dict_)
        for egoID in self.n_ego_dict.keys():
            self.n_ego_dict[egoID]['v_x'] = ego_v_x = n_ego_dict_[egoID]['v_x']
            self.n_ego_dict[egoID]['v_y'] = ego_v_y = n_ego_dict_[egoID]['v_y']
            self.n_ego_dict[egoID]['r'] = ego_r = n_ego_dict_[egoID]['r']
            self.n_ego_dict[egoID]['x'] = ego_x = n_ego_dict_[egoID]['x']
            self.n_ego_dict[egoID]['y'] = ego_y = n_ego_dict_[egoID]['y']
            self.n_ego_dict[egoID]['phi'] = ego_phi = n_ego_dict_[egoID]['phi']

            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi,
                                                                                          self.n_ego_dict[egoID]['l'])
            egdeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            keeproute = 2
            try:
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            except traci.exceptions.TraCIException:
                if 'ego' not in traci.vehicle.getIDList():
                    traci.simulationStep()
                    traci.vehicle.addLegacy(vehID=egoID, routeID='dl',
                                            # depart=0, pos=20, lane=lane, speed=ego_dict['v_x'],
                                            typeID='self_car')
                    print(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
                    traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x ** 2 + ego_v_y ** 2))

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_others.items():
            ego_x = self.n_ego_dict[egoID]['x']
            ego_y = self.n_ego_dict[egoID]['y']
            ego_phi = self.n_ego_dict[egoID]['phi']
            ego_l = self.n_ego_dict[egoID]['l']
            ego_w = self.n_ego_dict[egoID]['w']
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = (ego_x + cos(ego_phi / 180 * pi) * ego_lw)
            ego_y0 = (ego_y + sin(ego_phi / 180 * pi) * ego_lw)
            ego_x1 = (ego_x - cos(ego_phi / 180 * pi) * ego_lw)
            ego_y1 = (ego_y - sin(ego_phi / 180 * pi) * ego_lw)
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh['x'] - ego_x) < 10 and fabs(veh['y'] - ego_y) < 10:
                    surrounding_lw = (veh['l'] - veh['w']) / 2
                    surrounding_x0 = (veh['x'] + cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y0 = (veh['y'] + sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_x1 = (veh['x'] - cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y1 = (veh['y'] - sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    collision_check_dis = ((veh['w'] + ego_w) / 2 + 0.5) ** 2
                    if (ego_x0 - surrounding_x0) ** 2 + (ego_y0 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x0 - surrounding_x1) ** 2 + (ego_y0 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x1) ** 2 + (ego_y1 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x0) ** 2 + (ego_y1 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True

        self.n_ego_collision_flag = flag_dict


def test_traffic():
    import numpy as np
    from dynamics_and_models import ReferencePath

    def _reset_init_state():
        ref_path = ReferencePath('straight')
        random_index = int(np.random.random() * (900 + 500)) + 700
        x, y, phi = ref_path.indexs2points(random_index)
        v = 8 * np.random.random()
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x.numpy(),
                             y=y.numpy(),
                             phi=phi.numpy(),
                             l=4.8,
                             w=2.2,
                             routeID='du',
                             ))

    init_state = dict(ego=dict(v_x=8., v_y=0, r=0, x=-30, y=1.5, phi=180, l=4.8, w=2.2, routeID='dl', ))
    # init_state = _reset_init_state()
    traffic = Traffic(100., mode='training', init_n_ego_dict=init_state, training_task='left')
    traffic.init_traffic(init_state)
    traffic.sim_step()
    for i in range(100000000):
        # for j in range(50):
        # traffic.set_own_car(init_state)
        # traffic.sim_step()
        # init_state = _reset_init_state()
        # traffic.init_traffic(init_state)
        traffic.sim_step()


if __name__ == "__main__":
    test_traffic()
