import os
import sys

# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import warnings
warnings.filterwarnings('ignore')

import json
import yaml
import argparse
import time
import pathlib

from src.train import *
from src.dataset import *
from src.utils import *
from config import *


def save_execution_time(log_dir, total_time):
    file_path = os.path.join(log_dir, 'execution_time.txt')
    f = open(file_path, 'w')
    f.write(str(total_time))
    f.close()


def main(cfg, save_dir, should_eval):
    seed_init(seed=cfg.seed)
    if args.action == 'train':
        
        print('--- Train Phase ---')
        
        train_dataset = TrainDataset(cfg, 'train')
        train_loader  = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_worker) 
        val_dataset   = TrainDataset(cfg, 'valid')
        val_loader    = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_worker)
        
        data_loader   = {'train':train_loader, 'valid':val_loader}
        
        trainer = Trainer(data_loader, cfg, should_eval)

        start_training_time = time.time()
        trainer.train()
        total_training_time = time.time() - start_training_time
        save_execution_time(save_dir, total_training_time)
        
        # print('--- Test Phase ---')
        # seed_init(seed=cfg.seed)
        # tester = Tester(cfg)
        # tester.test(set_type='test')
        #
        # if cfg.logging:
        #     neptune.stop()

    else:
        print('--- Test Phase ---')
        tester = Tester(cfg)
        tester.test(set_type='test')


def get_english_data(directory: str):
    return f'./config/EnglishData/base-EnglishData-{directory}.yaml', f'../../Models/EnglishData-{directory}/triann', directory != '2spks'



if __name__ == "__main__":
    cfg, save_dir, should_eval = get_english_data('2spks')

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    parser.add_argument('--config', default=cfg, help='config yaml file')
    parser.add_argument('--num_worker', type=int, default=0, help='Num workers')
    parser.add_argument('--seed', type=int, default=1234, help='seed number')
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    parser.add_argument('--logging', type=bool, default=False, help='Logging option')
    parser.add_argument('--resume', type=bool, default=False, help='Resume option')
    parser.add_argument('--checkpoint', type=str, default=save_dir, help='Results save path')
    parser.add_argument('--model_name', type=str, default='model-best.pth', help='Best model name')
    parser.add_argument('--n_uttr', type=int, default=1, help='Number of target utterances') # default:1 for a fair comparison
    
    args = parser.parse_args()
    cfg  = Config(args.config)
    cfg  = set_experiment(args, cfg) # merge arg and cfg, make directories
    print(cfg)
    main(cfg, save_dir, should_eval)
