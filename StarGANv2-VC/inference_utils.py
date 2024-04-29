import os
import json

ENGLISH_DATA_SEEN_PATH = '../../Data/EnglishData/test-seen'
ENGLISH_DATA_UNSEEN_PATH = '../../Data/EnglishData/test-unseen'


def get_english_data(spks: int):
    model_checkpoint = f'../../Models/EnglishData-{spks}spks/stargan/'

    with open(os.path.join(model_checkpoint, 'speaker_mapping.txt')) as f:
        data = f.read()
        data = data.replace('\'', '\"')
        speaker_label_mapping = json.loads(data)
        # speaker_label_mapping = ast.literal_eval(data)
    where_to_save_samples = f'../../samples/EnglishData-{spks}spks/stargan/'
    config_path = os.path.join(model_checkpoint, f'config-EnglishData-{spks}spks.yml')
    model_path = os.path.join(model_checkpoint, 'final.pth')
    jdc_model_path = 'Utils/JDC/bst.t7'
    vocoder_path = 'Vocoder/checkpoint-400000steps.pkl'

    return (
        config_path,
        model_path,
        jdc_model_path,
        vocoder_path,
        ENGLISH_DATA_SEEN_PATH,
        ENGLISH_DATA_UNSEEN_PATH,
        where_to_save_samples,
        speaker_label_mapping,
        True
    )


POLISH_DATA_SEEN_PATH = '../../Data/PolishData/test-seen'
POLISH_DATA_UNSEEN_PATH = '../../Data/PolishData/test-unseen'


def get_polish_data(spks: int):
    model_checkpoint = f'../../Models/PolishData-{spks}spks/stargan/'

    with open(os.path.join(model_checkpoint, 'speaker_mapping.txt')) as f:
        data = f.read()
        data = data.replace('\'', '\"')
        speaker_label_mapping = json.loads(data)
        # speaker_label_mapping = ast.literal_eval(data)
    where_to_save_samples = f'../../samples/PolishData-{spks}spks/stargan/'
    config_path = os.path.join(model_checkpoint, f'config-PolishData-{spks}spks.yml')
    model_path = os.path.join(model_checkpoint, 'final.pth')
    jdc_model_path = 'Utils/JDC/bst.t7'
    vocoder_path = 'Vocoder/checkpoint-400000steps.pkl'

    return (
        config_path,
        model_path,
        jdc_model_path,
        vocoder_path,
        POLISH_DATA_SEEN_PATH,
        POLISH_DATA_UNSEEN_PATH,
        where_to_save_samples,
        speaker_label_mapping,
        False
    )
