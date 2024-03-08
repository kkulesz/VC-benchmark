import os
import shutil


def main():
    POLISH_DATA_DIR_PATH = "../../../Data-raw/pl-asr-bigos/data"
    speakers = ['clarin-pjatk-mobile-15==0001', 'clarin-pjatk-mobile-15==0002', 'clarin-pjatk-mobile-15==0003',
                'clarin-pjatk-mobile-15==0004', 'clarin-pjatk-mobile-15==0005', 'clarin-pjatk-mobile-15==0006',
                'clarin-pjatk-studio-15==0001', 'clarin-pjatk-studio-15==0002', 'clarin-pjatk-studio-15==0003',
                'clarin-pjatk-studio-15==0004', 'clarin-pjatk-studio-15==0005', 'clarin-pjatk-studio-15==0006',
                'fair-mls-20==1889', 'fair-mls-20==8758', 'mailabs-19==0001', 'mailabs-19==0002',
                'pwr-azon-read-20==228', 'pwr-azon-read-20==229', 'pwr-azon-spont-20==99317',
                'pwr-azon-spont-20==99318', 'pwr-azon-spont-20==99319', 'pwr-azon-spont-20==99320',
                'pwr-maleset-unk==0001', 'pwr-shortwords-unk==0001', 'pwr-viu-unk==0001']
    speaker_dir_and_id = list(map(lambda s: tuple(s.split('==')), speakers))

    speaker_files = {}
    for speaker_dir, speaker_id_within_dir in speaker_dir_and_id:
        speaker_dir_full_path = os.path.join(POLISH_DATA_DIR_PATH, speaker_dir)
        wavs = [f for f in os.listdir(speaker_dir_full_path) if f.startswith(speaker_id_within_dir)]

        speaker_files[(speaker_dir, speaker_id_within_dir)] = wavs

    TARGET_DIR = "../../../Data/PolishData"
    os.makedirs(TARGET_DIR, exist_ok=True)
    for speaker_dir, speaker_id_within_dir in speaker_files.keys():
        final_speaker_id = f"{speaker_dir}~{speaker_id_within_dir}"
        speaker_dir_full_path = os.path.join(TARGET_DIR, final_speaker_id)
        os.makedirs(speaker_dir_full_path, exist_ok=True)
        for f in speaker_files[(speaker_dir, speaker_id_within_dir)]:
            src = os.path.join(POLISH_DATA_DIR_PATH, speaker_dir, f)
            dst = os.path.join(speaker_dir_full_path, f)
            shutil.copy(src, dst)


if __name__ == '__main__':
    main()
