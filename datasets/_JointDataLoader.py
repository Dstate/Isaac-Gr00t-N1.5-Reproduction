import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, DistributedSampler
from .UniEngine import ProcessorGroot, UniDataset, DefaultDataCollator


def build_joint_dataloader(dataset_path_list = ['assets/data/libero', 'assets/data/calvin', 'assets/data/simpler_bridge', 'assets/data/simpler_rt1'], # processor
                           batch_size=2, num_workers=2, # dataloader
                           shuffle=True, pin_mem=True, drop_last=True, # dataloader
                           world_size=1, global_rank=0, # dataloader
                           **kwargs):
    
    dataset_list = []
    for dataset_path in dataset_path_list:
        processor = ProcessorGroot(dataset_path)
        train_dataset = UniDataset(processor=processor)
        dataset_list.append(train_dataset)
    
    joint_dataset = ConcatDataset(dataset_list)
    sampler = DistributedSampler(joint_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(joint_dataset, batch_size=batch_size, num_workers=num_workers,
                                sampler=sampler, pin_memory=pin_mem, drop_last=drop_last, collate_fn=DefaultDataCollator())

    return train_dataloader
