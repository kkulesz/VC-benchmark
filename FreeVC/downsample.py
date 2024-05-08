import os
import argparse
import librosa
import numpy as np
from scipy.io import wavfile
import paths


def process(wav_name, args):
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = wav_name[:4]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '_mic2.flac' in wav_path:
        os.makedirs(os.path.join(args.out_dir1, speaker), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        save_name = wav_name.replace("_mic2.flac", ".wav")
        save_path1 = os.path.join(args.out_dir1, speaker, save_name)
        save_path2 = os.path.join(args.out_dir2, speaker, save_name)
        wavfile.write(
            save_path1,
            args.sr1,
            (wav1 * np.iinfo(np.int16).max).astype(np.int16)
        )
        wavfile.write(
            save_path2,
            args.sr2,
            (wav2 * np.iinfo(np.int16).max).astype(np.int16)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--sr2", type=int, default=22050, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default=paths.RAW_VCTK_PATH, help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default=paths.DOWNSAMPLED_16k_PATH, help="path to target dir")
    parser.add_argument("--out_dir2", type=str, default=paths.DOWNSAMPLED_22k_PATH, help="path to target dir")
    args = parser.parse_args()

    speakers = os.listdir(args.in_dir)
    speakers = speakers[:3]  # TODO: take only few for now
    for speaker in speakers:
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            files_in_speaker_dir = os.listdir(spk_dir)
            audio_in_speaker_dir = list(filter(lambda f: f.endswith(".flac"), files_in_speaker_dir))
            for f in audio_in_speaker_dir:
                process(f, args)
            print(f"{spk_dir} - finished")
