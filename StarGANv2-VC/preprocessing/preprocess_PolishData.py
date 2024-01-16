import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    all_files = os.listdir(data_path)
    speakers_dirs = [f for f in all_files if os.path.isdir(os.path.join(data_path, f))]
    return speakers_dirs


def main():
    RAW_CORPUS_PATH = os.path.join(os.getcwd(), '../../../Data/PolishData')
    OUTPUT_PATH = os.path.join(os.getcwd(), '../../../Data/PolishData-stargan')
    speakers = get_speakers(RAW_CORPUS_PATH)
    print(speakers)

    preprocess(RAW_CORPUS_PATH, OUTPUT_PATH, speakers)


if __name__ == '__main__':
    main()
