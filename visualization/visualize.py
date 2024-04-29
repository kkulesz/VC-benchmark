import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import itertools

MCD = 'MCD'
SNR = 'SNR'

STARGAN = 'stargan'
TRIANN = 'triann'

ENGLISH_LANG = 'english'
POLISH_LANG = 'polish'

SAVE_DIRECTORY = '../../Plots'
GENERATE_POLISH_LABELS = True
GENERATE_POLISH = False


def map_model_to_title(model: str):
    if model == STARGAN:
        return "StarGANv2-VC"
    elif model == TRIANN:
        return "TriANN-VC"
    else:
        raise "invalid model name"


def plot_metrics_models():
    import plot_metrics
    df = plot_metrics.read_results('../../samples')
    for metric, seen in itertools.product([MCD, SNR], [True, False]):
        if GENERATE_POLISH_LABELS:
            title = f'Wynik {metric} obliczony na wypowiedziach mówców {"obecnych" if seen else "nieobecnych"} w zbiorze treningowym'
            y_label = metric
            x_label = 'Liczba mówców w zbiorze treningowym'
        else:
            title = f'{metric} score on {"seen" if seen else "unseen"} speakers\' utterances over number of speakers in training dataset'
            y_label = metric
            x_label = 'Number of speakers in the training dataset'

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_metrics.filter_results_and_reformat_it_between_models(seen=seen, metric=metric, df=df,
                                                                       plot_title=title)
        sub_path = os.path.join('ang', 'between', 'seen' if seen else 'unseen')
        plot_file_path = os.path.join(SAVE_DIRECTORY, sub_path, f'{metric}_{"seen" if seen else "unseen"}_between_models.png')

        plot_metrics.bar_plot_of_metric_over_number_of_speakers_for_each_model(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path)


def plot_metrics_seen():
    import plot_metrics
    df = plot_metrics.read_results('../../samples')
    for metric, model in itertools.product([MCD, SNR], [STARGAN, TRIANN]):
        if GENERATE_POLISH_LABELS:
            title = f'Wynik {metric} modelu {map_model_to_title(model)} w zależności od obecności mówców testowych w zbiorze treningowym'
            y_label = metric
            x_label = 'Liczba mówców w zbiorze treningowym'
            bar_labels = ['widziani', 'niewidziani']
        else:
            title = f'{metric} score of {map_model_to_title(model)} on seen vs unseen speakers over number of speakers in training dataset'
            y_label = metric
            x_label = 'Number of speakers in the training dataset'
            bar_labels = ['seen', 'unseen']

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_metrics.filter_results_and_reformat_it_between_seen_and_unseen_speakers(model=model, metric=metric,
                                                                                         df=df, plot_title=title, bar_labels=bar_labels)

        plot_file_path = os.path.join(SAVE_DIRECTORY, 'ang', model, f'{metric}_{model}_seen_vs_unseen.png')

        plot_metrics.bar_plot_of_metric_over_number_of_speakers_for_each_model(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path)


def plot_execution_time():
    import plot_execution_time
    df = plot_execution_time.read_results('../../Models')
    if GENERATE_POLISH_LABELS:
        title = f'Czas treningu'
        y_label = 'Czas[s]'
        x_label = 'Liczba mówców w zbiorze treningowym'
    else:
        title = 'Training time'
        y_label = 'Time[s]'
        x_label = 'Number of speakers in the training dataset'

    plot_title, x_ticks, models_plot_properties = \
        plot_execution_time.filter_results_and_reformat_it(df=df, plot_title=title)
    plot_file_path = os.path.join(SAVE_DIRECTORY, 'ang', 'between', 'training_time.png')

    plot_execution_time.bar_plot_of_execution_time(plot_title, y_label, x_label, x_ticks, models_plot_properties, plot_file_path)


def plot_MOSNet_score_models(language: str):
    import plot_mosnet
    df = plot_mosnet.read_results('../../samples')
    for seen in [True, False]:
        if GENERATE_POLISH_LABELS:
            title = f'Wynik modelu MOSNet obliczony na wypowiedziach mówców {"obecnych" if seen else "nieobecnych"} w zbiorze treningowym'
            y_label = 'Wynik MOSNet'
            x_label = 'Liczba mówców w zbiorze treningowym'
            ground_truth_seen_label = 'pliki oryginalne'
            ground_truth_unseen_label = 'pliki oryginalnie'
        else:
            title = f'MOSNet score on {"seen" if seen else "unseen"} speakers\' utterances over number of speakers in training dataset'
            y_label = 'MOSNet score'
            x_label = 'Number of speakers in the training dataset'
            ground_truth_seen_label = 'original files'
            ground_truth_unseen_label = 'original files'

        if seen:
            with open(os.path.join('../../Data/EnglishData/test-seen', 'MOSnet_result_raw.txt')) as f:
                mosnet_seen_score = float(re.search(r'\d+.\d+', re.search(r'Average: \d+.\d+', f.read()).group()).group())
                ground_truth_seen = (mosnet_seen_score, ground_truth_seen_label)
                ground_truth_unseen = None
        else:
            with open(os.path.join('../../Data/EnglishData/test-unseen', 'MOSnet_result_raw.txt')) as f:
                mosnet_unseen_score = float(re.search(r'\d+.\d+', re.search(r'Average: \d+.\d+', f.read()).group()).group())
                ground_truth_seen = None
                ground_truth_unseen = (mosnet_unseen_score, ground_truth_unseen_label)

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_mosnet.filter_results_and_reformat_it_between_models(seen=seen, df=df, plot_title=title, language=language)

        sub_path = os.path.join('ang' if language == 'english' else 'pol', 'between', 'seen' if seen else 'unseen')
        plot_file_path = os.path.join(SAVE_DIRECTORY, sub_path, f'MOSNet_score{("_" + language) if language==POLISH_LANG else "" }_{"seen" if seen else "unseen"}_between_models.png')

        plot_mosnet.bar_plot_of_MOSNet_score_over_number_of_speakers(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path,
            ground_truth_seen=ground_truth_seen, ground_truth_unseen=ground_truth_unseen)


def plot_MOSNet_score_seen(language: str):
    import plot_mosnet
    df = plot_mosnet.read_results('../../samples')
    for model in [STARGAN, TRIANN]:
        if GENERATE_POLISH_LABELS:
            title = f'Wynik MOSnet modelu {map_model_to_title(model)} w zależności od obecności mówców testowych w zbiorze treningowym'
            y_label = 'Wynik MOSNet'
            x_label = 'Liczba mówców w zbiorze treningowym'
            bar_labels = ['widziani', 'niewidziani']
            ground_truth_seen_label = 'widziani - pliki oryginalne'
            ground_truth_unseen_label = 'niewidziani - pliki oryginalnie'
        else:
            title = f'MOSNet score of {map_model_to_title(model)} on seen vs unseen speakers over number of speakers in training dataset'
            y_label = 'MOSNet score'
            x_label = 'Number of speakers in the training dataset'
            bar_labels = ['seen', 'unseen']
            ground_truth_seen_label = 'seen - original files'
            ground_truth_unseen_label = 'unseen - original files'


        with open(os.path.join('../../Data/EnglishData/test-seen', 'MOSnet_result_raw.txt')) as f:
            mosnet_seen_score = float(re.search(r'\d+.\d+', re.search(r'Average: \d+.\d+', f.read()).group()).group())
            ground_truth_seen = (mosnet_seen_score, ground_truth_seen_label)
        with open(os.path.join('../../Data/EnglishData/test-unseen', 'MOSnet_result_raw.txt')) as f:
            mosnet_unseen_score = float(re.search(r'\d+.\d+', re.search(r'Average: \d+.\d+', f.read()).group()).group())
            ground_truth_unseen = (mosnet_unseen_score, ground_truth_unseen_label)

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_mosnet.filter_results_and_reformat_it_between_seen(model=model, df=df, plot_title=title, bar_labels=bar_labels, language=language)

        sub_path = os.path.join('ang' if language == 'english' else 'pol', model)
        plot_file_path = os.path.join(SAVE_DIRECTORY, sub_path, f'MOSNet_score{("_" + language) if language==POLISH_LANG else "" }_{model}_seen_vs_unseen.png')

        plot_mosnet.bar_plot_of_MOSNet_score_over_number_of_speakers(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path,
            ground_truth_seen=ground_truth_seen, ground_truth_unseen=ground_truth_unseen)


def remove_mosnet_results(directory: str = '../../samples'):
    files_pattern = os.path.join(directory, '*/*/*/*/MOSnet_result_raw.txt')
    files_to_remove = glob.glob(files_pattern)
    for f in files_to_remove:
        os.remove(f)

def create_dir_structure(save_dir):
    ang = os.path.join(save_dir, 'ang')
    pol = os.path.join(save_dir, 'pol')
    os.makedirs(ang, exist_ok=True)
    os.makedirs(pol, exist_ok=True)

    def create_dir_substructure(subdir):
        subdir_between = os.path.join(subdir, 'between')
        subdir_stargan = os.path.join(subdir, STARGAN)
        subdir_triann = os.path.join(subdir, TRIANN)
        os.makedirs(subdir_between, exist_ok=True)
        os.makedirs(subdir_stargan, exist_ok=True)
        os.makedirs(subdir_triann, exist_ok=True)

        subdir_between_seen = os.path.join(subdir_between, 'seen')
        subdir_between_unseen = os.path.join(subdir_between, 'unseen')
        os.makedirs(subdir_between_seen, exist_ok=True)
        os.makedirs(subdir_between_unseen, exist_ok=True)

    create_dir_substructure(ang)
    create_dir_substructure(pol)

def main():
    # stargan - deepskyblue
    # triann - lightsalmon

    # seen - palegreen
    # unseen - papayawhip

    create_dir_structure(SAVE_DIRECTORY)

    if not GENERATE_POLISH:
        plot_metrics_models()
        plot_metrics_seen()
        plot_execution_time()
        plot_MOSNet_score_models(ENGLISH_LANG)
        plot_MOSNet_score_seen(ENGLISH_LANG)
    if GENERATE_POLISH:
        plot_MOSNet_score_models(POLISH_LANG)
        plot_MOSNet_score_seen(POLISH_LANG)

    # remove_mosnet_results()


if __name__ == '__main__':
    main()
