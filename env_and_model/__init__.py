#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/2/17
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: __init__.py
# =====================================

from env_and_model.aircraft.aircraft_env import AircraftEnv, AircraftModel
from env_and_model.idc_real.endtoend import IdcRealEnv
from env_and_model.idc_real.dynamics_and_models import IdcRealModel
from env_and_model.idc_virtual.endtoend import IdcVirtualEnv
from env_and_model.idc_virtual.dynamics_and_models import IdcVirtualModel
from env_and_model.idc_virtual.e2e_planner.end2end import E2eEnv
from env_and_model.idc_virtual_veh.endtoend import IdcVirtualVehEnv
from env_and_model.idc_virtual_veh.dynamics_and_models import IdcVirtualVehModel
from env_and_model.path_tracking.path_tracking_env import PathTrackingEnv, PathTrackingModel
from env_and_model.multi_lane.multilane import MultiLane
from env_and_model.multi_lane.car_tracking.dynamics_and_models import MultiLaneModel

Name2EnvAndModelCls = dict(multi_lane=(MultiLane, MultiLaneModel))

