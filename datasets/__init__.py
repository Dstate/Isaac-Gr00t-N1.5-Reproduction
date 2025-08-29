from .UniEngine import build_uni_dataloader
from ._JointDataLoader import build_joint_dataloader

ENGINE_REPO = {
    'build_uni_dataloader': build_uni_dataloader,
    'build_joint_dataloader': build_joint_dataloader
}

def create_engine(engine_name, **kwargs):
    return ENGINE_REPO[engine_name](**kwargs)




from .UniEngine import build_uni_server

SERVER_REPO = {
    'build_uni_server': build_uni_server
}

def create_server(server_name, **kwargs):
    return SERVER_REPO[server_name](**kwargs)
