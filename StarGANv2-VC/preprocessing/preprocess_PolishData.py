import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    all_files = os.listdir(data_path)
    speakers_dirs = [f for f in all_files if os.path.isdir(os.path.join(data_path, f))]
    return speakers_dirs


def even_recs():
    input_dataset_path = os.path.join(os.getcwd(), '../../../Data/PolishData-0-original')
    output_path = os.path.join(os.getcwd(), '../../../Data/PolishData-1-even-recs')
    speakers = get_speakers(input_dataset_path)

    preprocess(input_dataset_path, output_path, speakers, save_txt=False)


def final_preprocess():
    input_dataset_path = os.path.join(os.getcwd(), '../../../Data/PolishData-2-splitted/TRAIN')
    output_path = os.path.join(os.getcwd(), '../../../Data/PolishData-STARGAN-TRAIN')
    speakers = get_speakers(input_dataset_path)

    preprocess(input_dataset_path, output_path, speakers, save_txt=True)


def main():
    # even_recs()
    final_preprocess()


if __name__ == '__main__':
    main()
