import os
import argparse
import torch
import librosa
from glob import glob

import utils
from wavlm.WavLM import WavLM, WavLMConfig


def process(filename):
    basename = os.path.basename(filename)
    speaker = basename[:4]
    save_dir = os.path.join(args.out_dir, speaker)
    os.makedirs(save_dir, exist_ok=True)
    wav, _ = librosa.load(filename, sr=args.sr)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav)
    save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
    torch.save(c.cpu(), save_name)


if __name__ == "__main__":
    data_dir = "../../Data/freevc-preprocessed"
    source_data_dir = f"{data_dir}/vctk-16k/"
    output_data_dir = f"{data_dir}/wavlm/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default=source_data_dir, help="path to input dir")
    parser.add_argument("--out_dir", type=str, default=output_data_dir, help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading WavLM for content...")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")
    
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    
    for filename in filenames:
        process(filename)
    