import os
from typing import List

from pydub import AudioSegment
from pydub.silence import split_on_silence

import pandas as pd
import random


def split(sound):
    dBFS = sound.dBFS
    chunks = split_on_silence(
        sound,
        min_silence_len=100,
        silence_thresh=dBFS - 16,
        keep_silence=100
    )
    return chunks


def combine(_src):
    audio = AudioSegment.empty()
    for i, filename in enumerate(os.listdir(_src)):
        # if filename.endswith('.wav'):
        #     filename = os.path.join(_src, filename)
        #     audio += AudioSegment.from_wav(filename)
        if filename.endswith('.flac') and \
                "mic1" in filename:  # recordings are duplicated. drop 'mic2' and take only 'mic1'
            filename = os.path.join(_src, filename)
            audio += AudioSegment.from_file(filename, "flac")
    return audio


def save_chunks(chunks, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    counter = 0

    target_length = 5 * 1000
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # if the last output chunk is longer than the target length,
            # we can start a new one
            output_chunks.append(chunk)

    for chunk in output_chunks:
        chunk = chunk.set_frame_rate(24000)
        chunk = chunk.set_channels(1)
        counter = counter + 1
        chunk.export(os.path.join(directory, str(counter) + '.wav'), format="wav")


def get_speakers(data_path: str) -> List[str]:
    speakers_dirs = os.listdir(data_path)
    speakers_dirs = list(filter(lambda s: "p" in s, speakers_dirs))
    return speakers_dirs


def downsample_to_25khz_and_save_to_wav(input_dir, output_dir, speaker):
    directory = os.path.join(output_dir, speaker)

    audio = combine(os.path.join(input_dir, speaker))
    chunks = split(audio)
    save_chunks(chunks, directory)


def get_df(output_dir, speakers):
    speakers = list(map(lambda s: int(s.replace("p", "")), speakers))

    data_list = []
    for path, subdirs, files in os.walk(output_dir):
        for name in files:
            # print(f"{path}  -  {name}")
            if name.endswith(".wav"):
                speaker = int(path.split('/')[-1].replace('p', ''))
                if speaker in speakers:
                    data_list.append({"Path": os.path.join(path, name), "Speaker": int(speakers.index(speaker)) + 1})
    return pd.DataFrame(data_list)

def save_in_proper_form(df, output_path, file_name):
    file_str = ""
    for index, k in df.iterrows():
        file_str += k['Path'] + "|" + str(k['Speaker'] - 1) + '\n'
    text_file = open(os.path.join(output_path, file_name), "w")
    text_file.write(file_str)
    text_file.close()

def shuffle_split_and_save_txts(output_path, df):
    df = df.sample(frac=1)
    split_idx = round(len(df) * 0.1)

    test_data = df[:split_idx]
    train_data = df[split_idx:]
    save_in_proper_form(test_data, output_path, "val_list.txt")
    save_in_proper_form(train_data, output_path, "train_list.txt")

def main():
    RAW_CORPUS_PATH = os.path.join(os.getcwd(), '../../../Data/VCTK/wav48_silence_trimmed')
    OUTPUT_PATH = os.path.join(os.getcwd(), '../../../Data/VCTK_preprocessed_stargan')
    speakers = get_speakers(RAW_CORPUS_PATH)
    print(speakers)

    for s in speakers:
        print(f"Preprocessing: {s}")
        downsample_to_25khz_and_save_to_wav(input_dir=RAW_CORPUS_PATH, output_dir=OUTPUT_PATH, speaker=s)
    df = get_df(output_dir=OUTPUT_PATH, speakers=speakers)
    shuffle_split_and_save_txts(OUTPUT_PATH, df)

if __name__ == '__main__':
    """
    NOTE: run from this directory (probably {project_root}/starGANv2-VC/preprocessing)
    """
    main()
