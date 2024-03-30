import os

import glob
import json
from statistics import mean
from typing import Tuple, List, Callable

from metrics.mcd import MCD
from metrics.snr import SNR


def get_dir_pairs(spks: int, model: str):
    assert model in ['triann', 'stargan']
    assert spks in [2, 5, 10, 25, 50]

    reference_dir_path = '../../Data/EnglishData/'
    converted_dir_path = f'../../samples/EnglishData-{spks}spks/{model}'

    import os.path as osp
    return (
        (osp.join(reference_dir_path, 'test-seen', 'VCC2SM1'),
         osp.join(converted_dir_path, 'seen', 'VCC2SF1_in_voice_of_VCC2SM1')),

        (osp.join(reference_dir_path, 'test-seen', 'VCC2SF1'),
         osp.join(converted_dir_path, 'seen', 'VCC2SM1_in_voice_of_VCC2SF1')),

        (osp.join(reference_dir_path, 'test-unseen', 'VCC2TM2'),
         osp.join(converted_dir_path, 'unseen', 'VCC2TF2_in_voice_of_VCC2TM2')),

        (osp.join(reference_dir_path, 'test-unseen', 'VCC2TF2'),
         osp.join(converted_dir_path, 'unseen', 'VCC2TM2_in_voice_of_VCC2TF2'))
    )


def get_paths(reference_path, converted_path):
    wav_reference = glob.glob(reference_path + '/*.wav')
    wav_converted = glob.glob(converted_path + '/*.wav')
    wav_reference.sort()
    wav_converted.sort()

    zipped = list(zip(wav_reference, wav_converted))
    for r, c in zipped:
        assert os.path.basename(r) == os.path.basename(c)

    return zipped


def get_all_results_over_directory(metric_func: Callable, paired: List[Tuple[str, str]]):
    results = []
    for r, c in paired:
        r = metric_func(r, c)
        # print(r)
        results.append(r)
    return mean(results)


def main():
    dir_pairs = get_dir_pairs(50, 'stargan')
    for reference_path, converted_path in dir_pairs:
        paired = get_paths(reference_path, converted_path)

        results = {}
        for metric_func in [MCD, SNR]:
            result = get_all_results_over_directory(metric_func, paired)
            results[metric_func.__name__] = result

        with open(os.path.join(converted_path, 'results.json'), 'w') as f:
            json.dump(results, f)
        print(results)


if __name__ == '__main__':
    main()
