from scipy.io import wavfile
import pysptk
from pysptk.synthesis import Synthesizer, MLSADF
import pyworld
import numpy as np
import math

fs = 16000
frame_period = 5.0
hop_length = int(fs * (frame_period * 0.001))
ms_fftlen = 4096


def _compute_static_features(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=5.0)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    return mc


def _modspec(x, n=4096, norm=None, return_phase=False):
    # DFT against time axis
    s_complex = np.fft.rfft(x, n=n, axis=0, norm=norm)
    assert s_complex.shape[0] == n // 2 + 1
    R, im = s_complex.real, s_complex.imag
    ms = R * R + im * im

    # TODO: this is ugly...
    if return_phase:
        return ms, np.exp(1.0j * np.angle(s_complex))
    else:
        return ms


def _mean_modspec(path):
    mgc = _compute_static_features(path)

    ms = np.log(_modspec(mgc, n=ms_fftlen))
    # print(ms)
    return ms
    # return np.mean(np.array(mss), axis=(0,))


def MSD(f1_path: str, f2_path: str):
    ms_into2out_f1 = _mean_modspec(f1_path)
    ms_into2out_f2 = _mean_modspec(f2_path)

    a = ms_into2out_f1[:].T
    b = ms_into2out_f2[:].T

    diff = np.mean(np.absolute(a - b))
    diff = (np.inner(diff, diff))

    msd = math.sqrt(1 / len(_mean_modspec(f1_path).T)) * math.sqrt(diff)
    return msd
