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
    for speaker_id, (src_path, speaker_idx, cnv_path) in speaker_dicts.items():
        if speaker_idx is None:
            print("unseen")
            speaker_idx = random.randint(0, len(speaker_dicts))
            label = torch.LongTensor([speaker_idx]).to(device)
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to(device), label)
        else:
            print("seen")
            wave, sr = librosa.load(src_path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = __preprocess(wave).to(device)

            with torch.no_grad():
                label = torch.LongTensor([speaker_idx])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[speaker_id] = (ref, label, cnv_path)

    return reference_embeddings


def display(dir_to_save_samples, wave, filename):
    from scipy.io.wavfile import write
    import os
    os.makedirs(dir_to_save_samples, exist_ok=True)
    write(os.path.join(dir_to_save_samples, filename), 24000, wave)


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


def convert(
        f0_model,
        starganv2,
        vocoder,
        source_wav_path,
        targets,
        root_dir_to_save_samples,
        spk_id_label_mapping,
        generate_debug_recs
):
    source_audio, _ = librosa.load(source_wav_path, sr=24000)
    source_audio = source_audio / np.max(np.abs(source_audio))
    source_audio.dtype = np.float32

    target_speakers_dict = {}
    for _, (spk_id_plus_rec, src_pth, cnv_path) in enumerate(targets):
        spk_id = spk_id_plus_rec.split('====')[0]
        label_idx = spk_id_label_mapping[spk_id]
        target_speakers_dict[spk_id_plus_rec] = (src_pth, label_idx, cnv_path)
    targets_reference_embeddings = __compute_style(starganv2, target_speakers_dict)
    start = time.time()

    source_audio = __preprocess(source_audio).to(device)
    converted_samples = {}
    reconstructed_samples = {}
    for key, (ref, _, cnv_path) in targets_reference_embeddings.items():
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

        converted_samples[key] = (y_out.numpy(), cnv_path)
        reconstructed_samples[key] = recon

    end = time.time()
    print('total processing time: %.3f sec\n\n' % (end - start))

    for target_speaker_id, (wave, cnv_path) in converted_samples.items():
        cnv_path_dir = os.path.dirname(cnv_path)
        cnv_path_filename = os.path.basename(cnv_path)

        display(cnv_path_dir, wave, cnv_path_filename)
        if generate_debug_recs:
            display(cnv_path_dir, reconstructed_samples[target_speaker_id], f'rec_{cnv_path_filename}')

    if generate_debug_recs:
        wave, sr = librosa.load(source_wav_path, sr=24000)
        mel = __preprocess(wave)
        c = mel.transpose(-1, -2).squeeze().to(device)
        with torch.no_grad():
            recon = vocoder.inference(c)
            recon = recon.view(-1).cpu().numpy()

        display(root_dir_to_save_samples, recon, f'reconstructed_source_by_vocoder.wav')
        display(root_dir_to_save_samples, wave, f'original.wav')


def get_stargan_demodata():
    model_checkpoint = '../../Models/stargan/demodata/'
    data_path = '../../Data/DemoData-2-splitted'
    speaker_label_mapping = {
        'p226': 10, 'p243': 14, 'p259': 7, 'p256': 11, 'p240': 9, 'p258': 13, 'p230': 12, 'p270': 8, 'p236': 5,
        'p231': 0, 'p229': 6, 'p233': 1, 'p232': 4, 'p227': 2, 'p244': 3}

    return (
        os.path.join(model_checkpoint, 'config-DemoData.yml'),
        os.path.join(model_checkpoint, 'final_00150_epochs.pth'),
        "Utils/JDC/bst.t7",
        "Vocoder/checkpoint-400000steps.pkl",
        os.path.join(data_path, 'TRAIN/p226/1.wav'),
        '../../Data/DemoData-2-splitted/TEST/SEEN',
        '../../samples/TEST-STARGAN',
        speaker_label_mapping
    )


def convert_whole_folder(
    f0_model,
    stargan,
    vocoder,
    data_path: str,
    save_dir_root: str,
    source_path: str,
    speaker_label_mapping
):
    os.makedirs(save_dir_root, exist_ok=True)
    targets = []
    for walk_root, dirs, files in os.walk(data_path):
        walk_root = os.path.normpath(walk_root)
        for d in dirs:
            relative_dir_path = walk_root[len(data_path):]
            dir_to_create = os.path.normpath(save_dir_root + relative_dir_path + '/' + d)
            os.makedirs(dir_to_create, exist_ok=True)
        if len(files) > 0:
            speaker_id = walk_root[len(os.path.dirname(walk_root)) + 1:]
            relative_dir_path = walk_root[len(data_path):]
            dir_to_save_converted_recs = save_dir_root + relative_dir_path
            for f in files:
                org_path = os.path.normpath(os.path.join(walk_root, f))
                cnv_path = os.path.normpath(os.path.join(dir_to_save_converted_recs, f))
                targets.append((f"{speaker_id}===={f}", org_path, cnv_path))

    convert(f0_model, stargan, vocoder, source_path, targets, save_dir_root, speaker_label_mapping,
            generate_debug_info=False)


def main():
    cfg_path, stargan_path, f0_model_path, vocoder_path, source_path, dir_of_recs_to_be_converted, root_dir_to_save_samples, speaker_label_mapping = get_stargan_demodata()

    f0_model = load_f0_model(f0_model_path)
    stargan = load_stargan(stargan_path, cfg_path)
    vocoder = load_vocoder(vocoder_path)

    convert_whole_folder(
        f0_model=f0_model, stargan=stargan, vocoder=vocoder,
        data_path=dir_of_recs_to_be_converted, save_dir_root=root_dir_to_save_samples, source_path=source_path,
        speaker_label_mapping=speaker_label_mapping
    )


if __name__ == '__main__':
    main()
