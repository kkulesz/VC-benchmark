import os.path
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import time

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

device = 'cuda'


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
    for speaker_id, (path, speaker_idx) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker_idx]).to(device)
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to(device), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = __preprocess(wave).to(device)

            with torch.no_grad():
                label = torch.LongTensor([speaker_idx])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[speaker_id] = (ref, label)

    return reference_embeddings


def display(dir_to_save_samples, wave, filename):
    from scipy.io.wavfile import write
    import os
    os.makedirs(dir_to_save_samples, exist_ok=True)
    print(filename)
    write(os.path.join(dir_to_save_samples, f"{filename}.wav"), 24000, wave)


def load_vocoder(vocoder_path: str):
    from parallel_wavegan.utils import load_model
    vocoder = load_model(vocoder_path).to(device).eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()
    return vocoder


def load_stargan(model_path: str, config_path: str):
    with open(config_path) as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = __build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to(device)
    starganv2.mapping_network = starganv2.mapping_network.to(device)
    starganv2.generator = starganv2.generator.to(device)
    return starganv2


# load F0 model
def load_f0_model(fo_model_path: str):
    f0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(fo_model_path)['net']
    f0_model.load_state_dict(params)
    _ = f0_model.eval()
    f0_model = f0_model.to(device)
    return f0_model


def convert(f0_model, starganv2, vocoder, source, targets, dir_to_save_samples):
    source_speaker_id, source_wav_path = source
    source_audio, _ = librosa.load(source_wav_path, sr=24000)
    source_audio = source_audio / np.max(np.abs(source_audio))
    source_audio.dtype = np.float32

    target_speakers_dict = {}  # speaker_id -> (path_to_speaker_wav_file, speaker_index)
    for idx, (spk, pth) in enumerate(targets):
        target_speakers_dict[spk] = (pth, idx)
    # speaker_id -> (speaker_embedding, speaker_index)
    targets_reference_embeddings = __compute_style(starganv2, target_speakers_dict)
    start = time.time()

    source_audio = __preprocess(source_audio).to(device)
    converted_samples = {}
    reconstructed_samples = {}
    for key, (ref, _) in targets_reference_embeddings.items():
        with torch.no_grad():
            f0_feat = f0_model.get_feature_GAN(source_audio.unsqueeze(1))
            stargan_out = starganv2.generator(source_audio.unsqueeze(1), ref, F0=f0_feat)

            stargan_out_reshaped = stargan_out.transpose(-1, -2).squeeze().to(device)
            y_out = vocoder.inference(stargan_out_reshaped)
            y_out = y_out.view(-1).cpu()

            raw_target_wave, _ = librosa.load(target_speakers_dict[key][0], sr=24000)
            raw_target_mel = __preprocess(raw_target_wave)
            raw_target_mel_reshaped = raw_target_mel.transpose(-1, -2).squeeze().to(device)
            recon = vocoder.inference(raw_target_mel_reshaped)
            recon = recon.view(-1).cpu().numpy()

        converted_samples[key] = y_out.numpy()
        reconstructed_samples[key] = recon

    end = time.time()
    print('total processing time: %.3f sec\n\n' % (end - start))

    for target_speaker_id, wave in converted_samples.items():
        target_speaker_dir = os.path.join(dir_to_save_samples, target_speaker_id)
        display(target_speaker_dir, wave, f"converted_{source_speaker_id}_to_{target_speaker_id}")
        display(target_speaker_dir, reconstructed_samples[target_speaker_id], f'reconstructed_only')

    wave, sr = librosa.load(source_wav_path, sr=24000)
    mel = __preprocess(wave)
    c = mel.transpose(-1, -2).squeeze().to(device)
    with torch.no_grad():
        recon = vocoder.inference(c)
        recon = recon.view(-1).cpu().numpy()

    display(dir_to_save_samples, recon, f'reconstructed_source_{source_speaker_id}_by_vocoder')
    display(dir_to_save_samples, wave, f'original_{source_speaker_id}')


def get_stargan_demodata():
    model_checkpoint = '../../Models\stargan\demodata/'
    data_path = '../../Data\DemoData-2-splitted'

    return (
        os.path.join(model_checkpoint, 'config-DemoData.yml'),
        os.path.join(model_checkpoint, 'final_00150_epochs.pth'),
        "Utils/JDC/bst.t7",
        "Vocoder/checkpoint-400000steps.pkl",
        ('p226', os.path.join(data_path, 'TRAIN/p226/1.wav')),
        [
            ('p233', os.path.join(data_path, 'TEST\SEEN\p233/5.wav')),
            ('p244', os.path.join(data_path, 'TEST\SEEN\p244/6.wav')),
            ('p256', os.path.join(data_path, 'TEST\SEEN\p256/29.wav'))
        ],
        "../../samples/stargan_demodata"
    )

def get_stargan_polishdata():
    model_checkpoint = '../../Models\stargan\polishdata'
    data_path = '../../Data\PolishData-2-splitted'

    return (
        os.path.join(model_checkpoint, 'config-PolishData.yml'),
        os.path.join(model_checkpoint, 'final_00150_epochs.pth'),
        "Utils/JDC/bst.t7",
        "Vocoder/checkpoint-400000steps.pkl",
        ('clarin-pjatk-studio-15~0002', os.path.join(data_path, 'TRAIN/clarin-pjatk-studio-15~0002/1.wav')),
        [
            ('clarin-pjatk-mobile-15~0001', os.path.join(data_path, 'TEST\SEEN\clarin-pjatk-mobile-15~0001/23.wav')),
            ('mailabs-19~0001', os.path.join(data_path, 'TEST\SEEN\mailabs-19~0001/14.wav')),
            ('pwr-azon-read-20~228', os.path.join(data_path, 'TEST\SEEN\pwr-azon-read-20~228/73.wav'))
        ],
        "../../samples/stargan_polishdata-150epochs"
    )

def get_stargan_polishdata_300epochs():
    model_checkpoint = '../../Models\stargan\polishdata-300epochs'
    data_path = '../../Data\PolishData-2-splitted'

    return (
        os.path.join(model_checkpoint, 'config-PolishData.yml'),
        os.path.join(model_checkpoint, 'final_00300_epochs.pth'),
        "Utils/JDC/bst.t7",
        "Vocoder/checkpoint-400000steps.pkl",
        ('clarin-pjatk-studio-15~0002', os.path.join(data_path, 'TRAIN/clarin-pjatk-studio-15~0002/1.wav')),
        [
            ('clarin-pjatk-mobile-15~0001', os.path.join(data_path, 'TEST\SEEN\clarin-pjatk-mobile-15~0001/23.wav')),
            ('mailabs-19~0001', os.path.join(data_path, 'TEST\SEEN\mailabs-19~0001/14.wav')),
            ('pwr-azon-read-20~228', os.path.join(data_path, 'TEST\SEEN\pwr-azon-read-20~228/73.wav'))
        ],
        "../../samples/stargan_polishdata-300epochs"
    )


def main():
    # cfg_path, stargan_path, f0_model_path, vocoder_path, source, targets, dir_to_save_samples = get_stargan_demodata()
    # cfg_path, stargan_path, f0_model_path, vocoder_path, source, targets, dir_to_save_samples = get_stargan_polishdata()
    cfg_path, stargan_path, f0_model_path, vocoder_path, source, targets, dir_to_save_samples = get_stargan_polishdata_300epochs()


    f0_model = load_f0_model(f0_model_path)
    stargan = load_stargan(stargan_path, cfg_path)
    vocoder = load_vocoder(vocoder_path)
    convert(f0_model, stargan, vocoder, source, targets, dir_to_save_samples)


if __name__ == '__main__':
    main()
