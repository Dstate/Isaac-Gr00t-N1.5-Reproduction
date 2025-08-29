import os
import os.path as osp
import torch
import json 
import numpy as np
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import h5py
import cv2
import torch
from typing import Any, Dict, Union
import traceback
import uvicorn
import json_numpy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from gr00t.model.transforms import GR00TTransform
from gr00t.data.schema import EmbodimentTag
from gr00t.model.transforms import DefaultDataCollator

class RandomCropRatio(A.ImageOnlyTransform):
    """Randomly crop a region of the input image according to a specified ratio.

    Args:
        ratio (float): Crop ratio relative to the original image size (0 < ratio <= 1).
        p (float): Probability of applying the transform. Default: 1.0.
        always_apply (bool | None): If True, always apply this transform.
    """

    def __init__(self, ratio: float = 0.95, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        if not 0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("ratio",)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_h, new_w = int(h*self.ratio), int(w*self.ratio)
        y_min = np.random.randint(0, h - new_h + 1)
        x_min = np.random.randint(0, w - new_w + 1)
        y_max = y_min + new_h
        x_max = x_min + new_w
        return img[y_min:y_max, x_min:x_max]


def build_base_transform(n_px, aug=True, 
                        crop_scale=0.95, crop_prob=1.0, 
                        jitter_prob=1.0, jitter_bright=0.3, 
                        jitter_contrast=0.4, jitter_saturation=0.5, 
                        jitter_hue=0.08):

    base_transform = []
    # augmentation and resize
    
    if aug:
        base_transform.append(RandomCropRatio(ratio=crop_scale, p=crop_prob))
        base_transform.append(A.ColorJitter(brightness=jitter_bright, contrast=jitter_contrast, 
                                            saturation=jitter_saturation, hue=jitter_hue, p=jitter_prob))
    else :
        base_transform.append(A.CenterCrop(height=int(n_px*crop_scale), width=int(n_px*crop_scale), p=crop_prob))    
        
    # build transform
    base_transform.append(A.Resize(height=n_px, width=n_px))
    base_transform = A.ReplayCompose(base_transform)
    return base_transform


class ProcessorGroot(object):
    def __init__(self, meta_file_path, training=True, eps=1e-6):
        assert osp.isfile(meta_file_path), 'dataset statistics don\'t exit'
        dataset_statistics = json.load(open(meta_file_path, 'r'))
        self.dataset_statistics = dataset_statistics
        self.action_max = np.array(dataset_statistics['action_max'])
        self.action_min = np.array(dataset_statistics['action_min'])
        self.proprio_max = np.array(dataset_statistics['proprio_max'])
        self.proprio_min = np.array(dataset_statistics['proprio_min'])
        self.img_transform = build_base_transform(224, aug=training)
        self.gr00t_transform = GR00TTransform(
            state_horizon=1, # fixed
            action_horizon=16, # fixed
            max_state_dim=64,
            max_action_dim=32,
        )
        self.embodiment_tag = EmbodimentTag("new_embodiment")
        self.gr00t_transform.embodiment_tag = self.embodiment_tag
        self.gr00t_transform._language_key = 'language'
        self.gr00t_transform.training = training

        # fix parameters
        self.eps = eps
        self.action_length = len(dataset_statistics['action_max'])
        self.proprio_length = len(dataset_statistics['proprio_max'])

    def preprocess_action(self, action):
        action = np.clip(action, a_max=self.action_max, a_min=self.action_min)
        action = (action - self.action_min) / (self.action_max - self.action_min + self.eps) * 2 - 1
        action = torch.from_numpy(action)
        return action
    
    def preprocess_proprio(self, proprio):
        proprio = torch.from_numpy(proprio)
        sin_state = torch.sin(proprio)
        cos_state = torch.cos(proprio)
        proprio = torch.cat([sin_state, cos_state], dim=-1)
        return proprio
    
    def preprocess_image(self, img, replay_params=None):
        if replay_params == None:
            transformed = self.img_transform(image=img)
            transformed_image = transformed['image']
            replay_params = transformed['replay']
        else :
            transformed = A.ReplayCompose.replay(replay_params, image=img)
            transformed_image = transformed['image']
        return transformed_image, replay_params

    def postprocess_action(self, action):
        # action B 16 32 -> B 16 7
        # B = tensor_flatten_action.shape[0]
        # action = tensor_flatten_action.reshape(B, -1, self.action_length)
        action = action[..., :self.action_length]
        action = action.to(torch.float32).numpy()
        action = (action + 1) / 2 * (self.action_max - self.action_min + self.eps) + self.action_min
        action = np.clip(action, a_max=self.action_max, a_min=self.action_min)
        return action

    def preprocess_groot(self, item):
        # item:
        #    'video' T V H W C
        #    'language' str
        #    'action' T(16) N
        #    'state'  T(1)  N
        item = self.gr00t_transform(item)
        return item
    

class UniDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor
        self.chunk_length = 16 # fixed
        self._load_metas()
    
    def _load_metas(self):
        dataset_statistics = self.processor.dataset_statistics
        traj_paths = dataset_statistics['traj_paths']
        self.obs_keys = dataset_statistics['obs_keys']
        self.lang_key = dataset_statistics['lang_key']
        self.action_key = dataset_statistics['action_key']
        self.proprio_key = dataset_statistics['proprio_key']
        self.control_type = dataset_statistics['control_type']
        self.metas = []
        for traj_path, traj_lengh in traj_paths:
            self.metas.extend([(traj_path, j) for j in range(traj_lengh)])

    def _load_from_raw_traj(self, traj_path, cur_idx):
        with h5py.File(traj_path, 'r') as f:
            # load images from all views
            raw_images = []
            for view in self.obs_keys:
                raw_img = cv2.imdecode(f[view][cur_idx], cv2.IMREAD_COLOR)
                raw_images.append(raw_img)
            # load actions with chunking
            np_action = f[self.action_key][()][cur_idx : cur_idx + self.chunk_length]
            if len(np_action) < self.chunk_length:
                cnt = self.chunk_length - len(np_action)
                if 'rel' in self.control_type:
                    padding = np.concatenate([
                        np.zeros(np_action.shape[1]-1),
                        [np_action[-1][-1]]
                    ])[None, ...].repeat(cnt, axis=0)
                elif 'abs' in self.control_type:
                    padding = np_action[-1:].repeat(cnt, axis=0)
                else:
                    raise NotImplementedError
                np_action = np.concatenate([np_action, padding], axis=0)
            # load proprio
            if self.proprio_key is None: 
                raw_proprio = np.zeros((1,7))
            else:
                raw_proprio = f[self.proprio_key][()][cur_idx : cur_idx + 1]
            
            # load instruction
            instruction = f[self.lang_key][()].decode('utf-8')
        return raw_images, np_action, raw_proprio, instruction

    def __len__(self):
        return len(self.metas) 
    
    def __getitem__(self, index):
        meta = self.metas[index]
        raw_images, np_action, raw_proprio, instruction = self._load_from_raw_traj(meta[0], meta[1])
       
        # proprio
        proprio = self.processor.preprocess_proprio(raw_proprio)
        actions = self.processor.preprocess_action(np_action)
        images = np.stack([self.processor.preprocess_image(img)[0] for img in raw_images])[None, ...]

        item = {
            'state': proprio,
            'action': actions,
            'video': images,
            'language': instruction
        }

        item = self.processor.preprocess_groot(item)
        return item

class UniAgent(object):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.to_batch = DefaultDataCollator()
        self.policy = None

    def set_policy(self, policy):
        assert hasattr(policy, 'generate') and callable(getattr(policy, 'generate')), \
        "The policy must have a callable 'generate' method."
        self.policy = policy
    
    # Now only support single sample
    def get_action(self, raw_images, raw_proprio, instruction): 
        # images [H W 3]
        # raw_proprio 9
        # instruction 'xxx'

        print(raw_proprio.shape)

        proprio = self.processor.preprocess_proprio(raw_proprio)[None, ...]
        images = np.stack([self.processor.preprocess_image(img)[0] for img in raw_images])[None, ...]
        
        item = {
            'state': proprio,
            'video': images,
            'language': instruction
        }

        item = self.processor.preprocess_groot(item)
        batch = self.to_batch([item])

        print(batch['state'].shape)
        
        action, _ = self.policy.generate(batch)
        action = self.processor.postprocess_action(action)
        return action[0]

    def infer(self, payload: Dict[str, Any]):
        # agent_view_images B H W 3
        # wrist_view_images B H W 3
        # raw_proprio B 9
        # instruction ['xxx', ..., 'xxx']
        # eval_horizon 600
        # t 0
        print('recieve a request')
        try:    
            raw_images = json_numpy.loads(payload["raw_images"])
            raw_proprio = json_numpy.loads(payload["raw_proprio"])
            instruction = payload["instruction"]
            
            pred_action = self.get_action(raw_images, raw_proprio, instruction)

            return JSONResponse(content={
                "pred_action": pred_action.tolist()
            })
        except:
            error_str = traceback.format_exc()
            print("Error occurred:", error_str)
            return JSONResponse(content={
                "pred_action": None,
                "error_str": error_str
            })

    def info_query(self):
        return JSONResponse(content={
            "save_path": self.save_path
        })

    def run(self, output_dir=None, ckpt_name=None, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        
        try:
            self.save_path = osp.join(output_dir, f"Eval_{ckpt_name}")
        except:
            self.save_path = None
        
        self.app.post("/act")(self.infer)
        self.app.post("/info")(self.info_query)
        uvicorn.run(self.app, host=host, port=port)

def build_uni_dataloader(meta_file_path, batch_size=2, num_workers=2, 
                        shuffle=True, pin_mem=True, drop_last=True, 
                        world_size=1, global_rank=0, **kwargs):
    
    processor = ProcessorGroot(meta_file_path)
    train_dataset = UniDataset(processor=processor)
    sampler = DistributedSampler(train_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last, collate_fn=DefaultDataCollator())
    return train_dataloader

def build_uni_server(meta_file_path, **kwargs):
    processor = ProcessorGroot(meta_file_path, training=False)
    agent = UniAgent(processor)
    return agent