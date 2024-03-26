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
import soundfile as sf
import torch
import kaldiio
import librosa
import resampy
import pyworld as pw

from src.train import *
from src.dataset import *
from src.utils import *
from src.vocoder import decode
from src.cpc import *
from config import *
from preprocess.spectrogram import logmelspectrogram
from model import TriAANVC

def normalize_lf0(lf0):      
    zero_idxs    = np.where(lf0 == 0)[0]
    nonzero_idxs = np.where(lf0 != 0)[0]
    if len(nonzero_idxs) > 0 :
        mean = np.mean(lf0[nonzero_idxs])
        std  = np.std(lf0[nonzero_idxs])
        if std == 0:
            lf0 -= mean
            lf0[zero_idxs] = 0.0
        else:
            lf0 = (lf0 - mean) / (std + 1e-8)
            lf0[zero_idxs] = 0.0
    return lf0    

def GetTestData(path, cfg):

    sr       = cfg.sampling_rate
    wav, fs  = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=cfg.top_db) # trim slience

    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
        fs  = sr
        
    assert fs == 16000, 'Downsampling needs to be done.'
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
        
    mel = logmelspectrogram(
                            x=wav,
                            fs=cfg.sampling_rate,
                            n_mels=cfg.n_mels,
                            n_fft=cfg.n_fft,
                            n_shift=cfg.n_shift,
                            win_length=cfg.win_length,
                            window=cfg.window,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax
                            )
    tlen         = mel.shape[0]
    frame_period = cfg.n_shift/cfg.sampling_rate*1000
    
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.sampling_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.sampling_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices])
    
    return wav, mel, lf0


def main(cfg, mel_stats_path, source, targets, save_dir, generate_debug_recs):
    seed_init(seed=cfg.seed)
    
    mel_stats = np.load(mel_stats_path)
    mean      = np.expand_dims(mel_stats[0], -1)
    std       = np.expand_dims(mel_stats[1], -1)
    
    output_list = []
    model       = TriAANVC(cfg.model.encoder, cfg.model.decoder).to(cfg.device)
    checkpoint  = torch.load(f'{cfg.checkpoint}/{cfg.model_name}', map_location=cfg.device)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        src_wav_np, src_mel, src_lf0_np = GetTestData(source, cfg.setting)
        if generate_debug_recs:
            feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=save_dir + '/feats.1'))
            feat_writer[os.path.join(save_dir, "source")] = src_mel
            feat_writer.close()
            decode(f'{save_dir}/feats.1.scp', save_dir, cfg.device)
            os.remove(f'{save_dir}/feats.1.scp')
            os.remove(f'{save_dir}/feats.1.ark')

        for org_path, cnv_path in targets:
            trg_wav_np, trg_mel, _ = GetTestData(org_path, cfg.setting)
            if cfg.train.cpc:
                cpc_model = load_cpc(f'{cfg.cpc_path}/cpc.pt').to(cfg.device)
                cpc_model.eval()
                with torch.no_grad():
                    src_wav  = torch.from_numpy(src_wav_np).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
                    trg_wav  = torch.from_numpy(trg_wav_np).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
                    src_feat = cpc_model(src_wav, None)[0].transpose(1,2)
                    trg_feat = cpc_model(trg_wav, None)[0].transpose(1,2)
            else:
                src_feat = (src_mel.T - mean) / (std + 1e-8)
                trg_feat = (trg_mel.T - mean) / (std + 1e-8)
                src_feat = torch.from_numpy(src_feat).unsqueeze(0).to(cfg.device)
                trg_feat = torch.from_numpy(trg_feat).unsqueeze(0).to(cfg.device)
            src_lf0 = torch.from_numpy(normalize_lf0(src_lf0_np)).unsqueeze(0).to(cfg.device)

            output = model(src_feat, src_lf0, trg_feat)
            output = output.squeeze(0).cpu().numpy().T * (std.squeeze(1) + 1e-8) + mean.squeeze(1)
            output_list.append([cnv_path, output, trg_mel])

            # Mel-spectrograms to Wavs
            cnv_path_dir = os.path.dirname(cnv_path)
            feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=cnv_path_dir+'/feats.1'))
            for (filename, output, trg_mel) in output_list:
                feat_writer[filename + '_cnv'] = output
                if generate_debug_recs:
                    feat_writer[filename + '_trg'] = trg_mel

            feat_writer.close()
            print('synthesize waveform...')
            decode(f'{cnv_path_dir}/feats.1.scp', save_dir, cfg.device)
            os.remove(f'{cnv_path_dir}/feats.1.scp')
            os.remove(f'{cnv_path_dir}/feats.1.ark')


def rename_converted_files(data_path):
    for walk_root, dirs, files in os.walk(data_path):
        cnv_files = list(filter(lambda f: f.endswith('cnv_gen.wav'), files))
        for f in cnv_files:
            from_ = os.path.join(walk_root, f)
            to_ = os.path.join(walk_root, f[:-len('_cnv_gen.wav')])
            os.rename(from_, to_)

        trg_files = list(filter(lambda f: f.endswith('trg_gen.wav'), files))
        for f in trg_files:
            from_ = os.path.join(walk_root, f)
            to_ = os.path.join(walk_root, 'rec_' + f[:-len('.wav_trg_gen.wav')] + '.wav')
            os.rename(from_, to_)

def convert_whole_folder(
        cfg,
        stats_path: str,
        data_path: str,
        save_dir_root: str,
        src_speaker_path: str
):
    os.makedirs(save_dir_root, exist_ok=True)
    # (original_file_path, converted_file_path)
    targets = []
    for walk_root, dirs, files in os.walk(data_path):
        walk_root = os.path.normpath(walk_root)
        for d in dirs:
            relative_dir_path = walk_root[len(data_path):]
            dir_to_create = os.path.normpath(save_dir_root + relative_dir_path + '/' + d)
            os.makedirs(dir_to_create, exist_ok=True)
        if len(files) > 0:
            relative_dir_path = walk_root[len(data_path):]
            dir_to_save_converted_recs = save_dir_root + relative_dir_path
            single_speaker_targets = []
            for f in files:
                org_path = os.path.normpath(os.path.join(walk_root, f))
                cnv_path = os.path.normpath(os.path.join(dir_to_save_converted_recs, f))
                single_speaker_targets.append((org_path, cnv_path))
            targets.append(single_speaker_targets)

    for single_speaker_targets in targets:
        main(cfg=cfg, mel_stats_path=stats_path, source=src_speaker_path, targets=single_speaker_targets, save_dir=save_dir_root,
             generate_debug_recs=True)
    rename_converted_files(save_dir_root)


def get_stargan_demodata():
    model_checkpoint = '../../Models/triann/demodata/'

    return (
        os.path.join(model_checkpoint, 'base-DemoData.yaml'),
        os.path.join(model_checkpoint, 'model-best.pth'),  # model-500.pth
        os.path.join(model_checkpoint, 'mel_stats.npy'),
        os.path.join('../../Data/DemoData-2-splitted/', 'TRAIN/p226/1.wav'),
        "../../samples/TEST-TRIANN",
        '../../Data/DemoData-2-splitted/TEST/SEEN'
    )


if __name__ == "__main__":
    cfg, model, mel_stats_path, src_path, save_dir, data_path = get_stargan_demodata()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=cfg, help='config yaml file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    # parser.add_argument('--sample_path', type=str, default='./samples', help='Sample path')
    # parser.add_argument('--src_name', type=str, nargs='+', default=['src.flac'], help='Sample source name')
    # parser.add_argument('--trg_name', type=str, nargs='+', default=['trg.flac'], help='Sample target name')

    parser.add_argument('--checkpoint', type=str, default=save_dir, help='Results load path')
    parser.add_argument('--model_name', type=str, default=model, help='Best model name')
    parser.add_argument('--seed', type=int, default=1234, help='Seed')
    
    args = parser.parse_args()
    cfg  = Config(args.config)
    cfg  = set_experiment(args, cfg)
    # print(cfg)

    convert_whole_folder(
        cfg=cfg,
        stats_path=mel_stats_path,
        data_path=data_path,
        save_dir_root=save_dir,
        src_speaker_path=src_path)
