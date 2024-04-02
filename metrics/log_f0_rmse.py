import pyworld
import numpy as np
import math
import librosa


def _world_encode_data(wav, fs, frame_period=5.0, coded_dim=24, num_mcep=24):
    wav = wav.astype(np.float64)
    f0, _ = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    return np.ma.log(f0)


def _log_f0_rmse(x, y):
    if len(x) < len(y):
        frame_len = len(x)
    else:
        frame_len = len(y)

    log_spec_dB_const = 1 / frame_len
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


def LOG_F0_RMSE(f1_path: str, f2_path: str):
    sr = 16000

    wav_f1, _ = librosa.load(f1_path, sr=sr, mono=True)
    wav_f2, _ = librosa.load(f2_path, sr=sr, mono=True)

    f0_f1 = _world_encode_data(wav_f1, fs=sr)
    f0_f2 = _world_encode_data(wav_f2, fs=sr)

    min_cost, _ = librosa.sequence.dtw(f0_f1[:].T, f0_f2[:].T, metric=_log_f0_rmse)
    return np.mean(min_cost)
