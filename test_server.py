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

class UniAgent(object):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.to_batch = DefaultDataCollator()
        self.policy = None
        
        traj_root = '/home/ldx/Reproduction/Gr00t/assets/data/simpler_bridge/hdf5'
        files = os.listdir(traj_root)
        for file in files:
            traj_path = os.path.join(traj_root, file)
            with h5py.File(traj_path, 'r') as f:
                lang = f['language_instruction'][()].decode('utf-8')
                action = f['action'][()]
            print(lang)
            if lang == 'put the spoon on top of the orange towel':
                self.action = action
                break

        print(len(self.action))


    def set_policy(self, policy):
        assert hasattr(policy, 'generate') and callable(getattr(policy, 'generate')), \
        "The policy must have a callable 'generate' method."
        self.policy = policy
    
    # Now only support single sample
    def get_action(self, raw_images, raw_proprio, instruction): 
        # images [H W 3]
        # raw_proprio 9
        # instruction 'xxx'
        
        
        res = self.action[:16]
        self.action = self.action[16:]
        if len(res) < 16:
            res = np.zeros((16, 7))
        print(res)
        return res

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


if __name__ == '__main__':
    agent = UniAgent(processor=None)
    agent.run(port=20200)