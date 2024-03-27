import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    speakers_dirs = os.listdir(data_path)
    speakers_dirs = list(filter(lambda s: os.path.isdir(os.path.join(data_path, s)), speakers_dirs))
    return speakers_dirs


def even_recs():
    input_dataset_path = os.path.join(os.getcwd(), '../../../Data/VCTK-40spks/raw')
    output_path = os.path.join(os.getcwd(), '../../../Data/VCTK-40spks/preprocessed')
    speakers = get_speakers(input_dataset_path)

    preprocess(input_dataset_path, output_path, speakers, save_txt=False)


def main():
    even_recs()


if __name__ == '__main__':
    main()
