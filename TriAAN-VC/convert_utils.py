import os
import json

ENGLISH_DATA_SEEN_PATH = '../../Data/EnglishData/test-seen'
ENGLISH_DATA_UNSEEN_PATH = '../../Data/EnglishData/test-unseen'


def get_english_data(spks: int):
    model_checkpoint = f'../../Models/EnglishData-{spks}spks/triann'
    where_to_save_samples = f'../../samples/EnglishData-{spks}spks/triann'

    return (
        os.path.join(model_checkpoint, f'base-EnglishData-{spks}spks.yaml'),
        os.path.join(model_checkpoint, 'model-last.pth'),
        os.path.join(model_checkpoint, 'mel_stats.npy'),
        ENGLISH_DATA_SEEN_PATH,
        ENGLISH_DATA_UNSEEN_PATH,
        where_to_save_samples
    )
