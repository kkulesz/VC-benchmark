import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

import pandas as pd


def __split(sound):
    dBFS = sound.dBFS
    chunks = split_on_silence(
        sound,
        min_silence_len=100,
        silence_thresh=dBFS - 16,
        keep_silence=100
    )
    return chunks


def __combine(_src):
    audio = AudioSegment.empty()
    for i, filename in enumerate(os.listdir(_src)):
        if filename.endswith('.wav'):
            filename = os.path.join(_src, filename)
            audio += AudioSegment.from_wav(filename)
        elif filename.endswith('.flac') and \
                "mic1" in filename:  # recordings are duplicated. drop 'mic2' and take only 'mic1'
            filename = os.path.join(_src, filename)
            audio += AudioSegment.from_file(filename, "flac")
    return audio


def __save_chunks(chunks, directory):
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


def __downsample_to_25khz_and_save_to_wav(input_dir, output_dir, speaker):
    directory = os.path.join(output_dir, speaker)

    audio = __combine(os.path.join(input_dir, speaker))
    chunks = __split(audio)
    __save_chunks(chunks, directory)


def __get_df(output_dir, speakers):
    data_list = []
    for path, subdirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith(".wav"):
                speaker = os.path.basename(path)
                if speaker in speakers:
                    data_list.append({"Path": os.path.join(path, name), "Speaker": int(speakers.index(speaker)) + 1})
    return pd.DataFrame(data_list)


def __save_in_proper_form(df, output_path, file_name):
    file_str = ""
    speaker_mapping = {}
    for index, k in df.iterrows():
        file_str += k['Path'] + "|" + str(k['Speaker'] - 1) + '\n'

        speaker_id = os.path.basename(os.path.dirname(os.path.normpath(k['Path'])))
        speaker_mapping[speaker_id] = k['Speaker'] - 1

    text_file = open(os.path.join(output_path, file_name), "w")
    text_file.write(file_str)
    text_file.close()

    spk_mapping_file = open(os.path.join(output_path, 'speaker_mapping.txt'), "w")
    spk_mapping_file.write(str(speaker_mapping))
    spk_mapping_file.close()


def __shuffle_split_and_save_txts(output_path, df, save_txt: bool):
    df = df.sample(frac=1)
    split_idx = round(len(df) * 0.1)

    test_data = df[:split_idx]
    train_data = df[split_idx:]
    if save_txt:
        __save_in_proper_form(test_data, output_path, "val_list.txt")
        __save_in_proper_form(train_data, output_path, "train_list.txt")


def preprocess(input_dir, output_dir, speakers, save_txt: bool):
    for s in speakers:
        print(f"Preprocessing: {s}")
        __downsample_to_25khz_and_save_to_wav(input_dir=input_dir, output_dir=output_dir, speaker=s)
    df = __get_df(output_dir=output_dir, speakers=speakers)
    __shuffle_split_and_save_txts(output_dir, df, save_txt)
