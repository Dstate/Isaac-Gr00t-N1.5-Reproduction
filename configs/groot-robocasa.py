# training
logger_type = "tensorboard"
save_interval = 10000
log_interval = 10
num_iters = 50000
learning_rate = 1e-4
weight_decay = 0
warm_steps = 500
eta_min_lr = 1e-4

# data engine
engine_name = "build_uni_dataloader"
dataset_path = "assets/data/RoboCasa"
batch_size=16
num_workers=16

# model settings
model_name = "build_gr00t_finetune"
tune_llm = True

# agent settings
server_name = "build_uni_server"
