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
import pathlib

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
    for speaker_id, (target_rec_path, speaker_idx, converted_rec_path) in speaker_dicts.items():
        if speaker_idx is None:
            print("unseen")
            speaker_idx = random.randint(0, len(speaker_dicts))
            label = torch.LongTensor([speaker_idx]).to(device)
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to(device), label)
        else:
            print("seen")
            wave, sr = librosa.load(target_rec_path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = __preprocess(wave).to(device)

            with torch.no_grad():
                label = torch.LongTensor([speaker_idx])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[speaker_id] = (ref, label, converted_rec_path)

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
        src_path,
        targets,
        root_dir_to_save_samples,
        speaker_label_mapping,
        generate_debug_recs
):
    source_audio, _ = librosa.load(src_path, sr=24000)
    source_audio = source_audio / np.max(np.abs(source_audio))
    source_audio.dtype = np.float32

    src_speaker_id = os.path.basename(os.path.dirname(src_path))

    target_speakers_dict = {}
    for _, (target_rec_path) in enumerate(targets):
        target_spk_id = os.path.basename(os.path.dirname(target_rec_path))
        speaker_idx = speaker_label_mapping[target_spk_id] if target_spk_id in speaker_label_mapping else None

        converted_rec_dir = os.path.join(root_dir_to_save_samples, f'{src_speaker_id}_in_voice_of_{target_spk_id}')
        pathlib.Path(converted_rec_dir).mkdir(parents=True, exist_ok=True)
        converted_rec_path = os.path.join(converted_rec_dir, os.path.basename(target_rec_path))
        target_speakers_dict[target_spk_id] = (target_rec_path, speaker_idx, converted_rec_path)

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
    print('total processing time: %.3f sec' % (end - start))

    for target_speaker_id, (wave, cnv_path) in converted_samples.items():
        cnv_path_dir = os.path.dirname(cnv_path)
        cnv_path_filename = os.path.basename(cnv_path)

        display(cnv_path_dir, wave, cnv_path_filename)
        if generate_debug_recs:
            display(cnv_path_dir, reconstructed_samples[target_speaker_id], f'rec_{cnv_path_filename}')

    # if generate_debug_recs:
    #     wave, sr = librosa.load(src_path, sr=24000)
    #     mel = __preprocess(wave)
    #     c = mel.transpose(-1, -2).squeeze().to(device)
    #     with torch.no_grad():
    #         recon = vocoder.inference(c)
    #         recon = recon.view(-1).cpu().numpy()
    #
    #     display(root_dir_to_save_samples, recon, f'reconstructed_source_by_vocoder.wav')
    #     display(root_dir_to_save_samples, wave, f'original.wav')



def convert_whole_folder(
    f0_model,
    stargan,
    vocoder,
    seen_dir: str,
    unseen_dir: str,
    where_to_save_samples: str,
    speaker_label_mapping
):
    pathlib.Path(where_to_save_samples).mkdir(parents=True, exist_ok=True)

    def get_rec_pairs_between_two_speakers_in_single_dir(directory: str):
        seen_spks = os.listdir(directory)
        assert len(seen_spks) == 2
        spk1_recs = sorted(os.listdir(os.path.join(directory, seen_spks[0])))
        spk1_recs = list(map(lambda f: os.path.join(directory, seen_spks[0], f), spk1_recs))
        spk2_recs = sorted(os.listdir(os.path.join(directory, seen_spks[1])))
        spk2_recs = list(map(lambda f: os.path.join(directory, seen_spks[1], f), spk2_recs))

        seen_recs = list(zip(spk1_recs, spk2_recs))
        for r1, r2 in seen_recs:
            assert os.path.basename(r1) == os.path.basename(r2)
        seen_recs_reversed = list(zip(spk2_recs, spk1_recs))

        return seen_recs + seen_recs_reversed

    total_seen_recs = get_rec_pairs_between_two_speakers_in_single_dir(seen_dir)
    total_unseen_recs = get_rec_pairs_between_two_speakers_in_single_dir(unseen_dir)

    for src, trg in total_seen_recs:
        convert(f0_model, stargan, vocoder, src, [trg], os.path.join(where_to_save_samples, 'seen'), speaker_label_mapping,
                generate_debug_recs=False)

    for src, trg in total_unseen_recs:
        convert(f0_model, stargan, vocoder, src, [trg], os.path.join(where_to_save_samples, 'unseen'), speaker_label_mapping,
                generate_debug_recs=False)


def main():
    import inference_utils
    cfg_path, stargan_path, f0_model_path, vocoder_path, seen_dir, unseen_dir, where_to_save_samples, speaker_label_mapping = inference_utils.get_english_data(10)

    f0_model = load_f0_model(f0_model_path)
    stargan = load_stargan(stargan_path, cfg_path)
    vocoder = load_vocoder(vocoder_path)

    convert_whole_folder(
        f0_model=f0_model, stargan=stargan, vocoder=vocoder,
        seen_dir=seen_dir, unseen_dir=unseen_dir, where_to_save_samples=where_to_save_samples,
        speaker_label_mapping=speaker_label_mapping
    )


if __name__ == '__main__':
    main()
