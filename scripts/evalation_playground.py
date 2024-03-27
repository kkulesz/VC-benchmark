import math
import numpy as np
import librosa
import os
import os.path as osp
import glob
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pypesq import pesq
import pyworld


SAMPLING_RATE = 16000
data_path = '../../Data-raw/Evaluation_playground'
original_dir = osp.join(data_path, 'SM1')
converted_dir = osp.join(data_path, 'converted_SM2_Speech_in_SM1_Voice')


def my_snr(original, converted):
    a = math.sqrt(np.mean(original ** 2))
    b = math.sqrt(np.mean(converted ** 2))
    return 10 * np.log10(a / b)


def my_mcd(original, converted, sr):
    mfcc1 = librosa.feature.mfcc(y=original, sr=sr)
    mfcc2 = librosa.feature.mfcc(y=converted, sr=sr)

    # Align MFCCs using DTW
    _, path = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)

    # Calculate MCD
    mcd_sum = 0
    for pair in path:
        mcd_sum += np.sum((mfcc1[:, pair[0]] - mfcc2[:, pair[1]]) ** 2)

    mcd = (10 / np.log(10)) * np.sqrt(2 * mcd_sum / len(path))

    return mcd


def my_pesq(original, converted, sr):
    n = len(original) - len(converted)
    if n > 0:
        converted = np.hstack((converted, np.zeros(abs(n))))
    elif n < 0:
        original = np.hstack((original, np.zeros(abs(n))))

    return pesq(original, converted, sr)


def main():
    wav_original = glob.glob(original_dir + '/*')
    wav_converted = glob.glob(converted_dir + '/*')
    wav_original.sort()
    wav_converted.sort()

    zipped = list(zip(wav_original, wav_converted))
    snr_results = []
    mcd_results = []
    pesq_results = []
    for o, c in zipped:
        # print(o + '--' + c)
        wo, sr_o = librosa.load(o, sr=SAMPLING_RATE)
        wc, sr_c = librosa.load(c, sr=SAMPLING_RATE)
        snr_results.append(my_snr(wo, wc))
        mcd_results.append(my_mcd(wo, wc, sr_o))
        # pesq_results.append(my_pesq(wo, wc, sr_o))

    print(np.mean(snr_results))
    print(np.mean(mcd_results))
    # print(np.mean(pesq_results))


if __name__ == '__main__':
    main()
