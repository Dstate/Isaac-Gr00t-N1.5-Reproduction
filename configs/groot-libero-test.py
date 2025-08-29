# training
logger_type = "tensorboard"
save_interval = 2000
log_interval = 10
num_iters = 10000
learning_rate = 1e-4
weight_decay = 0
warm_steps = 500
eta_min_lr = 1e-4

# data engine
engine_name = "build_uni_dataloader"
dataset_path = "assets/data/libero"
batch_size=8
num_workers=8

# model settings
model_name = "build_gr00t_finetune"
tune_llm = False

# agent settings
server_name = "build_uni_server"
