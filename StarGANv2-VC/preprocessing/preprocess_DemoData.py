import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    speakers_dirs = os.listdir(data_path)
    speakers_dirs = list(filter(lambda s: "p" in s, speakers_dirs))
    return speakers_dirs


def even_recs():
    input_dataset_path = os.path.join(os.getcwd(), '../../../Data/DemoData-0-original')
    output_path = os.path.join(os.getcwd(), '../../../Data/DemoData-1-even-recs')
    speakers = get_speakers(input_dataset_path)

    preprocess(input_dataset_path, output_path, speakers, save_txt=False)


def final_preprocess():
    input_dataset_path = os.path.join(os.getcwd(), '../../../Data/DemoData-2-splitted/TRAIN')
    output_path = os.path.join(os.getcwd(), '../../../Data/DemoData-STARGAN-TRAIN')
    speakers = get_speakers(input_dataset_path)

    preprocess(input_dataset_path, output_path, speakers, save_txt=True)


def main():
    # even_recs()
    final_preprocess()


if __name__ == '__main__':
    main()
