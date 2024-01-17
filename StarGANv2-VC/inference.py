import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

speakers = [273]


def __preprocess(wave):
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def __build_model(model_params):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema


def __compute_style(starganv2, speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = __preprocess(wave).to('cuda')

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)

    return reference_embeddings


def display(wave, path):
    from scipy.io.wavfile import write
    import os
    dir_to_save_samples = "../../samples"
    os.makedirs(dir_to_save_samples, exist_ok=True)
    write(os.path.join(dir_to_save_samples, f"{path}.wav"), 24000, wave)


def load_vocoder():
    from parallel_wavegan.utils import load_model
    vocoder = load_model("Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()
    return vocoder


def load_stargan():
    model_path = '../../Models/demodata/epoch_00150.pth'

    with open('Models/config.yml') as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = __build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')
    return starganv2


# load F0 model
def load_f0_model():
    f0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load("Utils/JDC/bst.t7")['net']
    f0_model.load_state_dict(params)
    _ = f0_model.eval()
    f0_model = f0_model.to('cuda')
    return f0_model


def convert(f0_model, starganv2, vocoder):
    selected_speakers = [273]
    k = random.choice(selected_speakers)
    wav_path = 'Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav'
    audio, source_sr = librosa.load(wav_path, sr=24000)
    audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32

    speaker_dicts = {}
    for s in selected_speakers:
        k = s
        speaker_dicts['p' + str(s)] = ('Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav', speakers.index(s))
    reference_embeddings = __compute_style(starganv2, speaker_dicts)

    import time
    start = time.time()

    source = __preprocess(audio).to('cuda:0')
    keys = []
    converted_samples = {}
    reconstructed_samples = {}
    converted_mels = {}

    for key, (ref, _) in reference_embeddings.items():
        with torch.no_grad():
            f0_feat = f0_model.get_feature_GAN(source.unsqueeze(1))
            out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

            c = out.transpose(-1, -2).squeeze().to('cuda')
            y_out = vocoder.inference(c)
            y_out = y_out.view(-1).cpu()

            if key not in speaker_dicts or speaker_dicts[key][0] == "":
                recon = None
            else:
                wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                mel = __preprocess(wave)
                c = mel.transpose(-1, -2).squeeze().to('cuda')
                recon = vocoder.inference(c)
                recon = recon.view(-1).cpu().numpy()

        converted_samples[key] = y_out.numpy()
        reconstructed_samples[key] = recon

        converted_mels[key] = out

        keys.append(key)
    end = time.time()
    print('total processing time: %.3f sec' % (end - start))

    for key, wave in converted_samples.items():
        print('Converted: %s' % key)
        display(wave, "Converted")
        print('Reference (vocoder): %s' % key)
        if reconstructed_samples[key] is not None:
            display(reconstructed_samples[key], "Reference (vocoder)")

    print('Original (vocoder):')
    wave, sr = librosa.load(wav_path, sr=24000)
    mel = __preprocess(wave)
    c = mel.transpose(-1, -2).squeeze().to('cuda')
    with torch.no_grad():
        recon = vocoder.inference(c)
        recon = recon.view(-1).cpu().numpy()
    display(recon, 'Original (vocoder)')
    print('Original:')
    display(wave, "Original")


def main():
    f0_model = load_f0_model()
    stargan = load_stargan()
    vocoder = load_vocoder()

    convert(f0_model, stargan, vocoder)


if __name__ == '__main__':
    main()
