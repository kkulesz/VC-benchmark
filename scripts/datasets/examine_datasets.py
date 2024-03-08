import os
import wave
from statistics import mean
import pandas as pd
from itertools import groupby

STARGAN_DEMO_DIR_PATH = "../../../Data/StarGANv2-VC"
POLISH_DATA_DIR_PATH = "../../../Data-raw/pl-asr-bigos/data"


def get_stargan_demo_data_speakers_dict():
    dir_path = STARGAN_DEMO_DIR_PATH
    speakers_dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

    speaker_files_dict = {}
    for speaker in speakers_dirs:
        speaker_dir_full_path = os.path.join(dir_path, speaker)
        speaker_wavs = [os.path.join(speaker_dir_full_path, f) for f in os.listdir(speaker_dir_full_path) if
                        f.endswith(".wav")]
        speaker_files_dict[speaker] = speaker_wavs

    return speaker_files_dict


def get_polish_data_speakers_dict():
    dir_path = POLISH_DATA_DIR_PATH
    subdirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

    speaker_files_dict = {}
    for sd in subdirs:
        subdir_full_path = os.path.join(dir_path, sd)
        wavs_in_subdir = [f for f in os.listdir(subdir_full_path) if f.endswith(".wav")]
        wavs_in_subdir.sort()

        grouped_per_speaker = [list(i) for j, i in groupby(wavs_in_subdir, lambda f: f.partition('-')[0])]
        for group in grouped_per_speaker:
            speaker_id = sd + '==' + group[0].partition('-')[0]
            group_fill_path = list(map(lambda f: os.path.join(dir_path, sd, f), group))
            speaker_files_dict[speaker_id] = group_fill_path

    # for k in (speaker_files_dict.keys()):
    #     print(k)
    return speaker_files_dict


def get_wavs_details(wav_files, speaker):
    durations = []
    for w in wav_files:
        with wave.open(w, "r") as audio_file:
            frame_rate = audio_file.getframerate()
            n_frames = audio_file.getnframes()
            durations.append(n_frames / float(frame_rate))

    return {
        "speaker": speaker,
        "number_of_wavs": len(wav_files),
        "total_length": round(sum(durations), 2),
        "min": round(min(durations), 2),
        "max": round(max(durations), 2),
        "avg": round(mean(durations), 2)
    }


def print_dataset_statistics(dataset_df):
    avg_number_of_wavs = dataset_df['number_of_wavs'].mean()
    min_number_of_wavs = dataset_df['number_of_wavs'].min()
    max_number_of_wavs = dataset_df['number_of_wavs'].max()
    shortest_wav = dataset_df['min'].min()
    longest_wav = dataset_df['max'].max()
    avg_total_length = dataset_df['total_length'].mean()
    min_total_length = dataset_df['total_length'].min()
    max_total_length = dataset_df['total_length'].max()

    print(f"number of speakers: \t{len(dataset_df)}")
    print(f"average number of recs: {round(avg_number_of_wavs)}")
    print(f"min number of recs: \t{round(min_number_of_wavs)}")
    print(f"max number of recs: \t{round(max_number_of_wavs)}")
    print(f"shortest rec: \t\t\t{round(shortest_wav, 2)}")
    print(f"longest rec: \t\t\t{round(longest_wav, 2)}")
    print(f"average rec length: \t{round(avg_total_length, 2)}")
    print(f"min rec length: \t\t{round(min_total_length, 2)}")
    print(f"max rec length: \t\t{round(max_total_length, 2)}")


def filter_speakers(df):
    # df = df[df['number_of_wavs'] > 75]
    df = df[df['total_length'] > 240]
    return df


def examine_dict(d):
    speakers_details = list(map(lambda k: get_wavs_details(d[k], k), d.keys()))
    speakers_details_df = pd.DataFrame.from_records(speakers_details)
    speakers_details_df = filter_speakers(speakers_details_df)
    print_dataset_statistics(speakers_details_df)
    print(speakers_details_df['speaker'].to_list())


def main():
    # speakers_dict = get_stargan_demo_data_speakers_dict()
    speakers_dict = get_polish_data_speakers_dict()

    examine_dict(speakers_dict)


if __name__ == '__main__':
    main()
