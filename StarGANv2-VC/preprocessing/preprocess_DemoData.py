import os
from typing import List

from preprocessing_common import preprocess


def get_speakers(data_path: str) -> List[str]:
    speakers_dirs = os.listdir(data_path)
    speakers_dirs = list(filter(lambda s: "p" in s, speakers_dirs))
    return speakers_dirs


def main():
    RAW_CORPUS_PATH = os.path.join(os.getcwd(), '../../../Data/DemoData')
    OUTPUT_PATH = os.path.join(os.getcwd(), '../../../Data/DemoData-stargan')
    speakers = get_speakers(RAW_CORPUS_PATH)
    print(speakers)

    preprocess(RAW_CORPUS_PATH, OUTPUT_PATH, speakers)


if __name__ == '__main__':
    main()
