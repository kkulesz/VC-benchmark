import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def MCD(f1_path: str, f2_path: str) -> float:
    y1, sr1 = librosa.load(f1_path)
    y2, sr2 = librosa.load(f2_path)

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)

    # Align MFCCs using DTW
    _, path = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)

    # Calculate MCD
    mcd_sum = 0
    for pair in path:
        mcd_sum += np.sum((mfcc1[:, pair[0]] - mfcc2[:, pair[1]]) ** 2)

    mcd = (10 / np.log(10)) * np.sqrt(2 * mcd_sum / len(path))

    return mcd
