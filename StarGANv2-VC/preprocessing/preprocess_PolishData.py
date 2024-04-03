import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    all_files = os.listdir(data_path)
    speakers_dirs = [f for f in all_files if os.path.isdir(os.path.join(data_path, f))]
    return speakers_dirs


def even_recs():
    input_dataset_path = os.path.join(os.getcwd(), '../../../Data/PolishData-splitted/train')
    output_path = os.path.join(os.getcwd(), '../../../Data/PolishData/train')
    speakers = get_speakers(input_dataset_path)

    preprocess(input_dataset_path, output_path, speakers, save_txt=False)


def final_preprocess(directory):
    input_dataset_path = os.path.join(os.getcwd(), f'../../../Data/PolishData/{directory}/raw')
    output_path = os.path.join(os.getcwd(), f'../../../Data/PolishData/{directory}/stargan')
    os.makedirs(output_path, exist_ok=True)
    speakers = get_speakers(input_dataset_path)
    print(speakers)

    preprocess(input_dataset_path, output_path, speakers, save_txt=True)


def main():
    # even_recs()
    dirs = ['5spks', '10spks', '25spks']
    for d in dirs:
        final_preprocess(d)


if __name__ == '__main__':
    main()
