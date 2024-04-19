import os
import wave
from statistics import mean
import soundfile as sf
import pandas as pd
from itertools import groupby

VCTK_PATH = "../../../Data-raw/VCTK\wav48_silence_trimmed"
VCC18_PATH = "../../../Data/VCC18/vcc2018_training"
# STARGAN_DEMO_DIR_PATH = "../../../Data/StarGANv2-VC"
POLISH_DATA_DIR_PATH = "../../../Data-raw/pl-asr-bigos/data"


def get_VCC18_data_speakers_dict():
    dir_path = VCC18_PATH
    speakers_dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    speaker_files_dict = {}
    for speaker in speakers_dirs:
        speaker_dir_full_path = os.path.join(dir_path, speaker)
        speaker_wavs = [os.path.join(speaker_dir_full_path, f) for f in os.listdir(speaker_dir_full_path)]
        print(speaker_wavs)
        speaker_files_dict[speaker] = speaker_wavs
    return speaker_files_dict, True


def get_VCTK_data_speakers_dict():
    dir_path = VCTK_PATH
    speakers_dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

    speaker_files_dict = {}
    for speaker in speakers_dirs:
        speaker_dir_full_path = os.path.join(dir_path, speaker)
        speaker_wavs = [os.path.join(speaker_dir_full_path, f) for f in os.listdir(speaker_dir_full_path) if
                        f.endswith("mic1.flac")]
        speaker_files_dict[speaker] = speaker_wavs

    return speaker_files_dict, False


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
    return speaker_files_dict, True


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

def get_flacs_details(flac_files, speaker):
    import struct
    def bytes_to_int(bytes: list) -> int:
        result = 0
        for byte in bytes:
            result = (result << 8) + byte
        return result
    durations = []

    # for w in flac_files:
    #     with wave.open(w, "r") as audio_file:
    #         frame_rate = audio_file.getframerate()
    #         n_frames = audio_file.getnframes()
    #         durations.append(n_frames / float(frame_rate))
    for flac in flac_files:
        with open(flac, 'rb') as f:
            if f.read(4) != b'fLaC':
                raise ValueError('File is not a flac file')
            header = f.read(4)
            while len(header):
                meta = struct.unpack('4B', header)  # 4 unsigned chars
                block_type = meta[0] & 0x7f  # 0111 1111
                size = bytes_to_int(header[1:4])

                if block_type == 0:  # Metadata Streaminfo
                    streaminfo_header = f.read(size)
                    unpacked = struct.unpack('2H3p3p8B16p', streaminfo_header)

                    samplerate = bytes_to_int(unpacked[4:7]) >> 4
                    sample_bytes = [(unpacked[7] & 0x0F)] + list(unpacked[8:12])
                    total_samples = bytes_to_int(sample_bytes)
                    duration = float(total_samples) / samplerate

                    durations.append(duration)
                    break
                header = f.read(4)

    return {
        "speaker": speaker,
        "number_of_wavs": len(flac_files),
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
    total_length = dataset_df['total_length'].sum()
    avg_total_length = dataset_df['total_length'].mean()
    min_total_length = dataset_df['total_length'].min()
    max_total_length = dataset_df['total_length'].max()

    print(f"number of speakers: \t{len(dataset_df)}")
    print(f"average number of recs: {round(avg_number_of_wavs)}")
    print(f"min number of recs: \t{round(min_number_of_wavs)}")
    print(f"max number of recs: \t{round(max_number_of_wavs)}")
    print(f"shortest rec: \t\t\t{round(shortest_wav, 2)}")
    print(f"longest rec: \t\t\t{round(longest_wav, 2)}")
    print(f"total length: \t\t\t{round(total_length, 2)}")
    print(f"average rec length: \t{round(avg_total_length, 2)}")
    print(f"min rec length: \t\t{round(min_total_length, 2)}")
    print(f"max rec length: \t\t{round(max_total_length, 2)}")


def filter_speakers(df):
    # df = df[df['number_of_wavs'] > 75]
    df = df[df['total_length'] > 240]
    return df


def examine_dict(d, is_wave):
    if is_wave:
        speakers_details = list(map(lambda k: get_wavs_details(d[k], k), d.keys()))
    else:
        speakers_details = list(map(lambda k: get_flacs_details(d[k], k), d.keys()))
    speakers_details_df = pd.DataFrame.from_records(speakers_details)
    print(speakers_details_df)
    # speakers_details_df = filter_speakers(speakers_details_df)
    print_dataset_statistics(speakers_details_df)
    print(speakers_details_df['speaker'].to_list())


def main():
    speakers_dict, is_wave = get_VCC18_data_speakers_dict()
    # speakers_dict, is_wave = get_VCTK_data_speakers_dict()
    # speakers_dict, is_wave = get_polish_data_speakers_dict()

    examine_dict(speakers_dict, is_wave)


if __name__ == '__main__':
    main()
