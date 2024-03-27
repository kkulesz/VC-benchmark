import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    speakers_dirs = os.listdir(data_path)
    speakers_dirs = list(filter(lambda s: os.path.isdir(os.path.join(data_path, s)), speakers_dirs))
    return speakers_dirs


def final_preprocess(directory: str):
    input_dataset_path = os.path.join(os.getcwd(), f'../../../Data/EnglishData/{directory}/raw')
    output_path = os.path.join(os.getcwd(), f'../../../Data/EnglishData/{directory}/stargan')
    os.makedirs(output_path, exist_ok=True)
    speakers = get_speakers(input_dataset_path)
    print(speakers)

    preprocess(input_dataset_path, output_path, speakers, save_txt=True)


def main():
    dirs = ['2spks', '5spks', '10spks', '25spks', '50spks']
    for d in dirs:
        final_preprocess(d)


if __name__ == '__main__':
    main()
