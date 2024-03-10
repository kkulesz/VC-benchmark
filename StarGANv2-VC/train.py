#!/usr/bin/env python3
#coding:utf-8

import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
warnings.simplefilter('ignore')

from functools import reduce
from munch import Munch
import time

from meldataset import build_dataloader
from optimizers import build_optimizer
from models import build_model
from trainer import Trainer
# from torch.utils.tensorboard import SummaryWriter

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True #

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)

def main(config_path):
    config = yaml.safe_load(open(config_path))
    print(f"epochs: {config['epochs']}")
    print(f"train_data: {config['train_data']}")
    print(f"pretrained_model: {config['pretrained_model']}")
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    # writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    stage = config.get('stage', 'star')
    fp16_run = config.get('fp16_run', False)

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=4,
                                        device=device)
    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device)

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    with open(ASR_config) as f:
            ASR_config = yaml.safe_load(f)
    ASR_model_config = ASR_config['model_params']
    ASR_model = ASRCNN(**ASR_model_config)
    params = torch.load(ASR_path, map_location='cpu')['model']
    ASR_model.load_state_dict(params)
    _ = ASR_model.eval()

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(F0_path, map_location='cpu')['net']
    F0_model.load_state_dict(params)

    # build model
    model, model_ema = build_model(Munch(config['model_params']), F0_model, ASR_model)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 2e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    _ = [model[key].to(device) for key in model]
    _ = [model_ema[key].to(device) for key in model_ema]
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['mapping_network']['max_lr'] = 2e-6
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict=scheduler_params_dict)

    trainer = Trainer(args=Munch(config['loss_params']), model=model,
                            model_ema=model_ema,
                            optimizer=optimizer,
                            device=device,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            logger=logger,
                            fp16_run=fp16_run)

    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    def get_log_results_str(results_dict: dict) -> str:
        k_v_tuples = [(k, v) for k, v in results_dict.items()]
        return " | ".join(list(map(lambda k_v: '%-5s: %.4f' % (k_v[0], k_v[1]), k_v_tuples)))

    def save_if_best(total_eval_loss: torch.Tensor, checkpoint_path: str, epoch: int) -> None:
        best_model_path = os.path.join(checkpoint_path, 'best.pth')
        best_result_path = os.path.join(checkpoint_path, 'best-result.txt')
        best_epoch_path = os.path.join(checkpoint_path, 'best-epoch.txt')
        total_eval_loss = total_eval_loss.item()

        def save_value_to_file(file_path, value):
            f = open(file_path, 'w')
            f.write(str(value))
            f.close()

        if os.path.exists(best_result_path):
            f = open(best_result_path)
            best_so_far = float(f.read())
            f.close()
            if total_eval_loss < best_so_far:
                print(f"NEW BEST RESULT={total_eval_loss}")
                trainer.save_checkpoint(best_model_path)
                save_value_to_file(best_result_path, total_eval_loss)
                save_value_to_file(best_epoch_path, epoch)
            else:
                print(f"WORSE LOSS THAN PREVIOUSLY={total_eval_loss}")
        else:
            print(f"NO BEST MODEL SAVED SO FAR - saving with the result={total_eval_loss}")
            trainer.save_checkpoint(best_model_path)
            save_value_to_file(best_result_path, total_eval_loss)
            save_value_to_file(best_epoch_path, epoch)


    for _ in range(1, epochs+1):
        epoch = trainer.epochs

        start_train = time.time()
        train_results = trainer._train_epoch()
        train_time = time.time() - start_train

        start_eval = time.time()
        eval_results, total_eval_loss = trainer._eval_epoch()
        eval_time = time.time() - start_eval

        logger.info(f'--- epoch {epoch} out of {epochs} ---')
        logger.info('train time: %f -- eval time: %f ' % (train_time, eval_time))
        logger.info(f"train: {get_log_results_str(train_results)}")
        logger.info(f"eval : {get_log_results_str(eval_results)}")

        save_if_best(total_eval_loss, log_dir, epoch)

        if (epoch % save_freq) == 0:
            print("Saving...")
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))
    trainer.save_checkpoint(osp.join(log_dir, 'final_%05d_epochs.pth' % epochs))
    return 0


def get_data_path_list(train_path, val_path):
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list


if __name__ == "__main__":
    # import os
    # torch.backends.cudnn.benchmark = True
    # torch.cuda.set_per_process_memory_fraction(1.0)
    # torch.cuda.empty_cache()
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
    start = time.time()
    main()
    print(f"Elapse time: {time.time() - start}")
