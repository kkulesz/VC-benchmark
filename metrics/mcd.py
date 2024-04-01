import librosa
import math
import os
import pyworld
import pysptk
import numpy as np
import shutil
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

_SAMPLING_RATE = 22050
_FRAME_PERIOD = 5.0


def _load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)

    return wav


def _MCD(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y

    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


def _MCEP(wav_file, save_directory, mcep_file, alpha=0.65, fft_size=512, mcep_size=24):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    loaded_wav_file = _load_wav(wav_file, sr=_SAMPLING_RATE)

    _, spectral_envelop, _ = pyworld.wav2world(loaded_wav_file.astype(np.double), fs=_SAMPLING_RATE,
                                               frame_period=_FRAME_PERIOD, fft_size=fft_size)

    mcep = pysptk.sptk.mcep(spectral_envelop, order=mcep_size, alpha=alpha, maxiter=0,
                            etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    np.save(os.path.join(save_directory, mcep_file),
            mcep,
            allow_pickle=False)


def MCD(f1_path: str, f2_path: str):
    tmp_mcep_save_dir = '../../TMP_MCEP'
    if os.path.exists(tmp_mcep_save_dir):
        shutil.rmtree(tmp_mcep_save_dir)

    f1_mcep_filename = os.path.basename(f1_path).split('.')[0] + '_1.npy'
    _MCEP(f1_path, tmp_mcep_save_dir, f1_mcep_filename)

    f2_mcep_filename = os.path.basename(f2_path).split('.')[0] + '_2.npy'
    _MCEP(f2_path, tmp_mcep_save_dir, f2_mcep_filename)

    min_cost_tot = 0.0
    total_frames = 0

    f1_mcep_npy = np.load(os.path.join(tmp_mcep_save_dir, f1_mcep_filename))
    frame_no = len(f1_mcep_npy)
    f2_mcep_npy = np.load(os.path.join(tmp_mcep_save_dir, f2_mcep_filename))

    min_cost, _ = librosa.sequence.dtw(f1_mcep_npy[:, 1:].T, f2_mcep_npy[:, 1:].T, metric=_MCD)

    min_cost_tot += np.mean(min_cost)

    total_frames += frame_no

    shutil.rmtree(tmp_mcep_save_dir)
    mcd = min_cost_tot / total_frames
    # return mcd, total_frames
    return mcd
