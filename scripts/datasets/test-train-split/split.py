import os
from typing import List, Dict, Tuple
import random
import shutil


def _save(
        dataset_path: str,
        unseen_spks: List[str],
        save_dir: str,
        seen_speakers_splits: Dict[str, Tuple[List[str], List[str]]]
) -> None:
    train_dir_name = "TRAIN"
    test_dir_name = "TEST"

    seen_dir_name = "SEEN"
    unseen_dir_name = "UNSEEN"

    train_dir = os.path.join(save_dir, train_dir_name)
    test_dir = os.path.join(save_dir, test_dir_name)
    seen_dir = os.path.join(test_dir, seen_dir_name)
    unseen_dir = os.path.join(test_dir, unseen_dir_name)
    """
    Directory structure
      Train
          {A-speakers}
      Test
          SEEN
              {A-speakers}
          UNSEEN
              {B-speakers}
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(seen_dir, exist_ok=True)
    os.makedirs(unseen_dir, exist_ok=True)

    for spk in seen_speakers_splits:
        train_split, test_split = seen_speakers_splits[spk]

        train_dst_dir = os.path.join(train_dir, spk)
        os.makedirs(train_dst_dir, exist_ok=True)
        for f in train_split:
            src = os.path.join(dataset_path, spk, f)
            dst = os.path.join(train_dst_dir, f)
            shutil.copyfile(src, dst)

        test_dst_dir = os.path.join(seen_dir, spk)
        os.makedirs(test_dst_dir, exist_ok=True)
        for f in test_split:
            src = os.path.join(dataset_path, spk, f)
            dst = os.path.join(test_dst_dir, f)
            shutil.copyfile(src, dst)

    for spk in unseen_spks:
        src = os.path.join(dataset_path, spk)
        dst = os.path.join(unseen_dir, spk)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def split(dataset_path: str, unseen_spks: List[str], train_ratio_prc: int, save_dir: str) -> None:
    """
    1. read all speakers
    2. split speakers into those present in train dataset and those not (seen/unseen)
    3. for each seen speaker:
      - shuffle recordings
      - split those into test/train
    4. save everything in proper directory
    """
    all_spks = list(filter(lambda f: os.path.isdir(os.path.join(dataset_path, f)), os.listdir(dataset_path)))
    seen_spks = list(filter(lambda s: s not in unseen_spks, all_spks))

    # make sure we don't lose any speakers,
    # and that we passed proper unseen speakers (those actually present in the dataset)
    assert len(seen_spks) + len(unseen_spks) == len(all_spks)

    seen_speakers_splits = {}
    for s in seen_spks:
        spk_recs = os.listdir(os.path.join(dataset_path, s))
        random.shuffle(spk_recs)

        split_element = int(len(spk_recs) * train_ratio_prc / 100)
        train_split = spk_recs[:split_element]
        test_split = spk_recs[split_element:]

        assert len(train_split) > 0
        assert len(test_split) > 0

        seen_speakers_splits[s] = (train_split, test_split)
    # print(seen_speakers_splits)
    _save(dataset_path, unseen_spks, save_dir, seen_speakers_splits)


def get_demodata_info():
    dataset_path = "../../../../Data/DemoData-1-even-recs"
    unseen_speakers = ['p225', 'p239', 'p228', 'p273', 'p254']
    speaker_train_ratio_perc = 95

    save_dir = "../../../../Data/DemoData-2-splitted"

    return dataset_path, unseen_speakers, speaker_train_ratio_perc, save_dir


def get_polishdata_info():
    dataset_path = "../../../../Data/PolishData-1-even-recs"
    unseen_speakers = ['clarin-pjatk-mobile-15~0003', 'clarin-pjatk-mobile-15~0004', 'clarin-pjatk-studio-15~0001', 'pwr-azon-spont-20~99317', 'fair-mls-20~8758']
    speaker_train_ratio_perc = 95

    save_dir = "../../../../Data/PolishData-2-splitted"

    return dataset_path, unseen_speakers, speaker_train_ratio_perc, save_dir


def main():
    # dataset_path, unseen_speakers, train_ratio_prc, save_dir = get_demodata_info()
    dataset_path, unseen_speakers, train_ratio_prc, save_dir = get_polishdata_info()

    split(dataset_path, unseen_speakers, train_ratio_prc, save_dir)


if __name__ == '__main__':
    main()
