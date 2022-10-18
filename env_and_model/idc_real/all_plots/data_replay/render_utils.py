import os
import re
import shutil
import subprocess

import magic
from protobuf_to_dict import protobuf_to_dict

import env_and_model.idc_real.all_plots.data_replay.idc_info_pb2 as pb

# policy_type
IS_MIX = True

# action
if IS_MIX:
    ACC_SCALE = 1.5
    ACC_SHIFT = 0.5
    STEER_SCALE = 0.3
    STEER_SHIFT = 0
else:
    ACC_SCALE = 2.25
    ACC_SHIFT = 0.75
    STEER_SCALE = 0.4
    STEER_SHIFT = 0

# controller
STEER_RATIO = 16.6

# plot option
SHOW_INTERESTED = False
SHOW_OTHER = False


def get_replay_data(try_path, start_time=0):
    filepaths = os.listdir(try_path)
    idc_planner_info_list = []
    for filepath in filepaths:
        if is_binwary_file(try_path+'/'+filepath):
            binfile = open(try_path+'/'+filepath, 'rb')  # 打开二进制文件
            size = os.path.getsize(try_path+'/'+filepath)  # 获得文件大小
            data = binfile.read(size)
            binfile.close()
            IdcPlannerInfo = pb.IdcPlannerInfo()
            IdcPlannerInfo.ParseFromString(data)
            d = protobuf_to_dict(IdcPlannerInfo)
            idc_planner_info_list.append(d)
    idc_planner_info_list = sorted(idc_planner_info_list, key=lambda x: x['timestamp'])
    return idc_planner_info_list[start_time:]


def is_binwary_file(ff):
    mime_kw = 'x-executable|x-sharedlib|octet-stream|x-object'
    try:
        magic_mime = magic.from_file(ff, mime=True)
        magic_hit = re.search(mime_kw, magic_mime, re.I)
        if magic_hit:
            return True
        else:
            return False
    except Exception as e:
        return False


def image2video(forder):
    os.chdir(forder)
    subprocess.call(['ffmpeg', '-framerate', '10', '-i', 'step%03d.png', 'video.mp4'])


def get_list_of_participants_in_obs(input_):
    other_start_dim = 21
    max_other_num = 18
    def get_list_of_participants_dict(msg):
        ego_x, ego_y = msg['ego_state']['x'], msg['ego_state']['y']
        selected_path_idx = msg['decision'].get('selected_path_idx', None)
        selected_path_idx = selected_path_idx if selected_path_idx else 0
        selected_obs = msg['obs_vector'][selected_path_idx]['input_vector']
        out = []
        for i in range(max_other_num):
            other_vector = selected_obs[other_start_dim + i * 10:other_start_dim + (i + 1) * 10]
            parti_dict = dict(zip(['x', 'y', 'v', 'phi', 'l', 'w'], other_vector[:6]))
            parti_dict.setdefault('type', other_vector[6:9])
            parti_dict['x'] += ego_x
            parti_dict['y'] += ego_y
            out.append(parti_dict)
        return out
    return list(map(lambda msg: get_list_of_participants_dict(msg), input_))


def get_list_of_path_values(input_):
    path_num = len(input_[0]['decision']['path_value'])
    out = []  # each element is a list, which includes the i-th path value during running
    for i in range(path_num):
        out.append(list(map(lambda msg: msg['decision']['path_value'][i], input_)))
    return out


def get_light_phase(case_idx):
    if case_idx in [16, 17, 31, 32]:
        return 3
    elif case_idx in [18, 19, 20, 21]:
        return 2
    else:
        return 0

def update_light_phase(case_idx, t):
    if case_idx in [16, 17, 31, 32]:
        return 4
    elif case_idx == 18:
        if t <= 2.5:
            return 0
        elif 2.5 < t < 6:
            return 2
        else:
            return 4
    elif case_idx == 21:
        if t <= 2:
            return 2
        else:
            return 4
    elif case_idx in [19, 20]:
        return 2
    else:
        return 0


def pt2inch(pt):
    return pt/72


def mycopyfile(srcfile, dstpath, name=None):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        fname = name if name is not None else fname
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))