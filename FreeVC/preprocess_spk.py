import os
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
import glob
import argparse


def build_from_path(in_dir, out_dir, weights_fpath):
    wavfile_paths = glob.glob(os.path.join(in_dir, '*.wav'))
    wavfile_paths = sorted(wavfile_paths)
    results = []
    for wav_path in wavfile_paths:
        res = _compute_spkEmbed(out_dir, wav_path, weights_fpath)
        results.append(res)
    return results


def _compute_spkEmbed(out_dir, wav_path, weights_fpath):
    utt_id = os.path.basename(wav_path).rstrip(".wav")
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)

    encoder = SpeakerEncoder(weights_fpath)
    embed = encoder.embed_utterance(wav)
    fname_save = os.path.join(out_dir, f"{utt_id}.npy")
    np.save(fname_save, embed, allow_pickle=False)
    return os.path.basename(fname_save)


def preprocess(in_dir, out_dir_root, spk, weights_fpath):
    out_dir = os.path.join(out_dir_root, spk)
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, weights_fpath)


if __name__ == "__main__":
    data_dir = "../../Data/freevc-preprocessed"
    source_data_dir = f"{data_dir}/vctk-16k/"
    output_data_dir = f"{data_dir}/vctk-16k-preprocessed_spk/"
    os.makedirs(output_data_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=source_data_dir)
    parser.add_argument('--out_dir_root', type=str, default=output_data_dir)
    parser.add_argument('--spk_encoder_ckpt', type=str, default='speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    args = parser.parse_args()

    sub_folder_list = os.listdir(args.in_dir)
    sub_folder_list.sort()

    ckpt_step = os.path.basename(args.spk_encoder_ckpt).split('.')[0].split('_')[-1]
    spk_embed_out_dir = os.path.join(args.out_dir_root, "spk")
    print("[INFO] spk_embed_out_dir: ", spk_embed_out_dir)
    os.makedirs(spk_embed_out_dir, exist_ok=True)

    for spk in sub_folder_list:
        print("Preprocessing {} ...".format(spk))
        in_dir = os.path.join(args.in_dir, spk)
        if not os.path.isdir(in_dir):
            continue
        # out_dir = os.path.join(args.out_dir, spk)
        preprocess(in_dir, spk_embed_out_dir, spk, args.spk_encoder_ckpt)

