import warnings

warnings.filterwarnings(action='ignore')
import os
from os.path import join as opj
import json
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import Config, Write_json, seed_init, MakeDir
from preprocess.audio import GetSpeakerInfo, SplitDataset, ProcessingTrainData, ExtractMelstats, SaveFeatures, \
    GetMetaResults, GetSpeakers


def main(cfg):
    seed_init()
    MakeDir(cfg.output_path)
    # all_spks, gen2spk = GetSpeakerInfo(cfg)
    all_spks = GetSpeakers(cfg)

    print('---Split dataset---')
    # all_wavs, train_wavs_names, valid_wavs_names, test_wavs_names = SplitDataset(all_spks, cfg)
    all_wavs, train_wavs_names, valid_wavs_names = SplitDataset(all_spks, cfg)

    print('---Feature extraction---')
    results = Parallel(n_jobs=-1)(delayed(ProcessingTrainData)(wav_path, cfg) for wav_path in tqdm(all_wavs))

    wn2info = {}
    for r in results:
        wav_path, mel, lf0, mel_len, speaker = r
        wav_path = os.path.normpath(wav_path)
        print(wav_path)
        wn2info[wav_path] = [mel, lf0, mel_len, speaker]
    print(len(wn2info))
    mean, std = ExtractMelstats(wn2info, train_wavs_names, cfg)  # only use train wav for normalizing stats

    print('---Write Features---')
    train_results = Parallel(n_jobs=-1)(delayed(SaveFeatures)(wav_name, wn2info[os.path.normpath(wav_name)], 'train', cfg) for wav_name in tqdm(train_wavs_names))
    valid_results = Parallel(n_jobs=-1)(delayed(SaveFeatures)(wav_name, wn2info[os.path.normpath(wav_name)], 'valid', cfg) for wav_name in tqdm(valid_wavs_names))
    # test_results  = Parallel(n_jobs=-1)(delayed(SaveFeatures)(wav_name, wn2info[wav_name], 'test', cfg) for wav_name in tqdm(test_wavs_names))

    # train_results, valid_results, test_results = GetMetaResults(train_results, valid_results, test_results, cfg)

    print('---Write Infos---')
    Write_json(train_results, f'{cfg.output_path}/train.json')
    Write_json(valid_results, f'{cfg.output_path}/valid.json')
    # Write_json(test_results, f'{cfg.output_path}/test.json')
    print('---Done---')


if __name__ == '__main__':
    # cfg = Config('./config/preprocess-DemoData.yaml')
    cfg = Config('./config/preprocess-PolishData.yaml')
    main(cfg)
