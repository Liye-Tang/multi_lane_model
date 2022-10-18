#!/bin/bash
# ['aircraft', 'idc_real', 'idc_virtual', 'inverted_pendulum',"
#  'path_tracking', 'suspension']
python3 ampc_script.py --env_id aircraft
python3 ampc_script.py --env_id inverted_pendulum
python3 ampc_script.py --env_id path_tracking
python3 ampc_script.py --env_id suspension