import os
import os.path as osp
from pathlib import Path
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from utils import check_hdf5_structure

def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--dataset_root', type=str)
    cfg = parser.parse_args()
    return cfg

def check_hdf5(dataset_path):
    hdf5_files = [str(file.resolve()) for file in Path(dataset_path).rglob('*.hdf5')]
    check_hdf5_structure(hdf5_files[0])

def build_dataset_statistics(dataset_path, cache_json='cache.json', obs_keys=['observation/third_image'], 
                             action_key='action', proprio_key='proprio', lang_key='language_instruction', control_type='rel_eef'):
    if osp.isfile(cache_json):
        print('dataset statistics exits')
        dataset_statistics = json.load(open(cache_json, 'r'))
    else :
        print('Beginning to build dataset statistics...')
        hdf5_files = [str(file.resolve()) for file in Path(dataset_path).rglob('*.hdf5')]
        traj_paths = []
        proprios = []
        actions = []
        # check all data
        for file in tqdm(hdf5_files):
            with h5py.File(file, 'r') as f:
                traj_length = len(f[action_key])
                traj_actions = f[action_key][()].astype('float32')
                traj_proprios = f[proprio_key][()].astype('float32') if proprio_key else np.zeros((traj_length, 7))
                actions.append(traj_actions)
                proprios.append(traj_proprios)
                traj_paths.append((file, traj_length))
                

        # calculate statistics
        actions = np.concatenate(actions, axis=0)
        proprios = np.concatenate(proprios, axis=0)
        action_max = np.quantile(actions, 0.99, axis=0).tolist()
        action_min = np.quantile(actions, 0.01, axis=0).tolist()
        proprio_max = np.quantile(proprios, 0.99, axis=0).tolist()
        proprio_min = np.quantile(proprios, 0.01, axis=0).tolist()
        dataset_statistics = dict(dataset_path=dataset_path,
                                  obs_keys=obs_keys, control_type=control_type,
                                  action_key=action_key, proprio_key=proprio_key, lang_key=lang_key,
                                  action_max=action_max, action_min=action_min,
                                  proprio_max = proprio_max, proprio_min = proprio_min, 
                                  traj_paths=traj_paths)
        with open(cache_json, 'w') as f:
            json.dump(dataset_statistics, f, indent=4)
    return dataset_statistics

def build_statistics(config):
    if 'libero' in config['dataset_root']:
        print('processing libero data...')
        obs_keys=['observation/third_image',
                'observation/wrist_image']
        
        action_key='action'
        proprio_key='proprio'
        control_type='rel_eef'
        build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/libero.json',
                                obs_keys=obs_keys, control_type=control_type,
                                action_key=action_key, proprio_key=proprio_key)
        print('ok')
    elif 'calvin' in config['dataset_root']:
        print('processing calvin data...')
        obs_keys=['observation/third_image',
                'observation/wrist_image']
        action_key='rel_action'
        proprio_key='proprio'
        control_type='rel_eef'
        build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/calvin.json',
                                 obs_keys=obs_keys, control_type=control_type,
                                action_key=action_key, proprio_key=proprio_key)
        print('ok')
    elif 'simpler' in config['dataset_root']:
        if 'bridge' in config['dataset_root']:
            print('processing simpler_bridge data...')
            obs_keys=['observation/third_image']
            action_key='action'
            proprio_key='proprio'
            control_type='rel_eef'
            build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/simpler_bridge.json',
                                     obs_keys=obs_keys, control_type=control_type,
                                    action_key=action_key, proprio_key=proprio_key)
            print('ok')
        elif 'rt1' in config['dataset_root']:
            print('processing simpler_bridge data...')
            obs_keys=['observation/image0']
            action_key='action'
            proprio_key=None
            control_type='rel_eef'
            build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/simpler_rt1.json',
                                     obs_keys=obs_keys, control_type=control_type,
                                    action_key=action_key, proprio_key=proprio_key)
            print('ok')
    elif 'VLABench' in config['dataset_root']:
        print('processing VLABench data...')
        obs_keys=['observation/image0',
                'observation/image2',
                'observation/wrist_image']
        action_key='action'
        proprio_key='proprio'
        control_type='abs_eef'
        build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/VLABench.json',
                                 obs_keys=obs_keys, control_type=control_type,
                                action_key=action_key, proprio_key=proprio_key)
        print('ok')
    elif 'RoboTwin' in config['dataset_root']:
        check_hdf5(config['dataset_root'])
        obs_keys=['observation/head_view',
                'observation/left_view',
                'observation/right_view']
        action_key='joint_action'
        proprio_key='proprio'
        control_type='abs_joint'
        build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/RoboTwin.json',
                                 obs_keys=obs_keys, control_type=control_type,
                                action_key=action_key, proprio_key=proprio_key)
    elif 'RoboCasa' in config['dataset_root']:
        print('processing robocasa data...')
        obs_keys=["obs/robot0_agentview_left_image",
                "obs/robot0_agentview_right_image",
                "obs/robot0_eye_in_hand_image"]
        action_key='actions_abs'
        proprio_key='obs/robot0_joint_pos'
        control_type='abs_eef'
        build_dataset_statistics(config['dataset_root'], cache_json='assets/metas/RoboCasa.json',
                                 obs_keys=obs_keys, control_type=control_type,
                                action_key=action_key, proprio_key=proprio_key)
        print('ok')
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # config = vars(get_args_parser())
    # check_hdf5(config['dataset_root'])
    # build_statistics(config)

    files = os.listdir('assets/data')
    for file in files:
        config = dict(dataset_root=os.path.join('assets/data', file))
        build_statistics(config)
        