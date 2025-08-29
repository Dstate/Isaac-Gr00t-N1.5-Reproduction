import os
import torch

from models import create_model
from datasets import create_server
from utils import Gr00tModelWrapper
import json
import argparse

def main(path, ckpt_name, strict_load=True, host="0.0.0.0", port=8000):
    json_file = os.path.join(path, 'config.json')
    config = json.load(open(json_file, 'r'))
    
    model = create_model(**config)
    ckpt_file = os.path.join(path, f"{ckpt_name}.pth")
    model.load_state_dict(torch.load(ckpt_file, map_location='cpu'), strict=strict_load)
    model = Gr00tModelWrapper(model)
    agent = create_server(**config)

    agent.set_policy(model)
    agent.run(path, ckpt_name, host=host, port=port)

def get_args_parser():
    parser = argparse.ArgumentParser('server model', add_help=False)
    parser.add_argument('--port', type=int)
    parser.add_argument('--device_id', type=int)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--ckpt_name', type=str)
    cfg = parser.parse_args()
    return cfg

if __name__ == '__main__':
    cfg = get_args_parser()
    torch.cuda.set_device(cfg.device_id)
    main(cfg.ckpt_path, cfg.ckpt_name, False, "0.0.0.0", cfg.port)
