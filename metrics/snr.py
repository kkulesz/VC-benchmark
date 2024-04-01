import math
import librosa
import numpy as np


def SNR(f1_path: str, f2_path: str):
    y1, _ = librosa.load(f1_path)
    y2, _ = librosa.load(f2_path)

    a = math.sqrt(np.mean(y1 ** 2))
    b = math.sqrt(np.mean(y2 ** 2))

    # snr = 10 * np.log10(a / b)
    snr = abs(10 * np.log10(a / b))

    return snr
