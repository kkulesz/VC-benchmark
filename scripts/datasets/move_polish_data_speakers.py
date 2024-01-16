import os
import shutil


def main():
    POLISH_DATA_DIR_PATH = "../../../Data-raw/pl-asr-bigos/data"
    speakers = ['fair-mls-20==1889', 'mailabs-19==0001', 'mailabs-19==0002', 'pwr-azon-read-20==228',
                'pwr-azon-read-20==229', 'pwr-maleset-unk==0001', 'pwr-shortwords-unk==0001']
    dir_spk_prefix = list(map(lambda s: tuple(s.split('==')), speakers))

    speaker_files = {}
    for speaker_dir, prefix in dir_spk_prefix:
        speaker_dir_full_path = os.path.join(POLISH_DATA_DIR_PATH, speaker_dir)
        wavs = [f for f in os.listdir(speaker_dir_full_path) if f.startswith(prefix)]

        speaker_files[speaker_dir] = wavs

    TARGET_DIR = "../../../Data/PolishData"
    os.makedirs(TARGET_DIR, exist_ok=True)
    for speaker_dir in speaker_files.keys():
        speaker_dir_full_path = os.path.join(TARGET_DIR, speaker_dir)
        os.makedirs(speaker_dir_full_path, exist_ok=True)
        for f in speaker_files[speaker_dir]:
            src = os.path.join(POLISH_DATA_DIR_PATH, speaker_dir, f)
            dst = os.path.join(TARGET_DIR, speaker_dir, f)
            shutil.copy(src, dst)


if __name__ == '__main__':
    main()
