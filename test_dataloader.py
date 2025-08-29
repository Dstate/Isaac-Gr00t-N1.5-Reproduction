
import os
from utils import check_dict_structure
from datasets import create_engine

files = os.listdir('assets/metas')
for file in files:
    dataset_path = os.path.join('assets/metas', file)
    train_loader = create_engine('build_uni_dataloader', meta_file_path=dataset_path)
    for batch in train_loader:
        check_dict_structure(batch)
        break