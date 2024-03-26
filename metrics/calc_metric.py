import os
from itertools import groupby
from typing import Callable, Tuple, List
from statistics import mean

from metrics.mcd_own_implementation.mcd import calculate_mcd


def calculate_metric(
        main_directory: str,
        paired_recordings: List[Tuple[str, List[Tuple[str, str]]]],
        metric_func: Callable[[str, str], float]
):
    speaker_results = []
    for speaker_dir, speaker_recs in paired_recordings:
        single_results = []
        for target_speaker_rec, converted_rec in speaker_recs:
            trg_path = os.path.join(main_directory, speaker_dir, target_speaker_rec)
            cvt_path = os.path.join(main_directory, speaker_dir, converted_rec)

            single_results.append((converted_rec, metric_func(trg_path, cvt_path)))
        speaker_mean = mean([t[1] for t in single_results])
        speaker_results.append((speaker_dir, speaker_mean, single_results))

    for s in speaker_results:
        print(s)

def pair_recordings(main_directory: str):
    result = []

    dirs = list(filter(lambda f: os.path.isdir(os.path.join(main_directory, f)), os.listdir(main_directory)))
    for d in dirs:
        wavs = sorted(list(filter(lambda f: f.endswith('.wav'), os.listdir(os.path.join(main_directory, d)))))

        assert len(wavs) % 2 == 0  # make sure there is (src, trg) pair for each recording

        orgs = []
        cvts = []
        for w in wavs:
            if w.startswith('rec_'):
                orgs.append(w)
            else:
                cvts.append(w)
        pairs = list(zip(orgs, cvts))
        result.append((d, pairs))

    return main_directory, result


def main():
    folder_path = '../../samples/TEST-TRIANN'
    main_dir, paired = pair_recordings(folder_path)
    calculate_metric(main_dir, paired, calculate_mcd)


if __name__ == '__main__':
    main()
