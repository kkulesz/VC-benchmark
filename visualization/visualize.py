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

SAVE_DIRECTORY = '../../Plots'
GENERATE_POLISH = True


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
        if GENERATE_POLISH:
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

        plot_file_path = os.path.join(SAVE_DIRECTORY, f'{metric}_{"seen" if seen else "unseen"}_between_models.png')

        plot_metrics.bar_plot_of_metric_over_number_of_speakers_for_each_model(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path)


def plot_metrics_seen():
    import plot_metrics
    df = plot_metrics.read_results('../../samples')
    for metric, model in itertools.product([MCD, SNR], [STARGAN, TRIANN]):
        if GENERATE_POLISH:
            title = f'Wynik {metric} modelu {map_model_to_title(model)} w zależności od obecności mówców testowych w zbiorze treningowym'
            y_label = metric
            x_label = 'Liczba mówców w zbiorze treningowym'
        else:
            title = f'{metric} score of {map_model_to_title(model)} on seen vs unseen speakers over number of speakers in training dataset'
            y_label = metric
            x_label = 'Number of speakers in the training dataset'

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_metrics.filter_results_and_reformat_it_between_seen_and_unseen_speakers(model=model, metric=metric,
                                                                                         df=df, plot_title=title)

        plot_file_path = os.path.join(SAVE_DIRECTORY, f'{metric}_{model}_seen_vs_unseen.png')

        plot_metrics.bar_plot_of_metric_over_number_of_speakers_for_each_model(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path)


def plot_execution_time():
    import plot_execution_time
    df = plot_execution_time.read_results('../../Models')
    if GENERATE_POLISH:
        title = f'Czas treningu'
        y_label = 'Czas[s]'
        x_label = 'Liczba mówców w zbiorze treningowym'
    else:
        title = 'Training time'
        y_label = 'Time[s]'
        x_label = 'Number of speakers in the training dataset'

    plot_title, x_ticks, models_plot_properties = \
        plot_execution_time.filter_results_and_reformat_it(df=df, plot_title=title)
    plot_file_path = os.path.join(SAVE_DIRECTORY, 'training_time.png')

    plot_execution_time.bar_plot_of_execution_time(plot_title, y_label, x_label, x_ticks, models_plot_properties, plot_file_path)


def plot_MOSNet_score_models():
    import plot_mosnet
    df = plot_mosnet.read_results('../../samples')
    for seen in [True, False]:
        if GENERATE_POLISH:
            title = f'Wynik modelu MOSNet obliczony na wypowiedziach mówców {"obecnych" if seen else "nieobecnych"} w zbiorze treningowym'
            y_label = 'Wynik MOSNet'
            x_label = 'Liczba mówców w zbiorze treningowym'
        else:
            title = f'MOSNet score on {"seen" if seen else "unseen"} speakers\' utterances over number of speakers in training dataset'
            y_label = 'MOSNet score'
            x_label = 'Number of speakers in the training dataset'

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_mosnet.filter_results_and_reformat_it_between_models(seen=seen, df=df, plot_title=title)

        plot_file_path = os.path.join(SAVE_DIRECTORY, f'MOSNet_score_{"seen" if seen else "unseen"}_between_models.png')

        plot_mosnet.bar_plot_of_MOSNet_score_over_number_of_speakers(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path)


def plot_MOSNet_score_seen():
    import plot_mosnet
    df = plot_mosnet.read_results('../../samples')
    for model in [STARGAN, TRIANN]:
        if GENERATE_POLISH:
            title = f'Wynik MOSnet modelu {map_model_to_title(model)} w zależności od obecności mówców testowych w zbiorze treningowym'
            y_label = 'Wynik MOSNet'
            x_label = 'Liczba mówców w zbiorze treningowym'
        else:
            title = f'MOSNet score of {map_model_to_title(model)} on seen vs unseen speakers over number of speakers in training dataset'
            y_label = 'MOSNet score'
            x_label = 'Number of speakers in the training dataset'

        plot_title, x_ticks, models_plot_properties, bottom = \
            plot_mosnet.filter_results_and_reformat_it_between_seen(model=model, df=df, plot_title=title)
        plot_file_path = os.path.join(SAVE_DIRECTORY, f'MOSNet_score_{model}_seen_vs_unseen.png')

        plot_mosnet.bar_plot_of_MOSNet_score_over_number_of_speakers(
            plot_title, y_label, x_label, x_ticks, models_plot_properties, bottom, plot_file_path)


def remove_mosnet_results(directory: str = '../../samples'):
    files_pattern = os.path.join(directory, '*/*/*/*/MOSnet_result_raw.txt')
    files_to_remove = glob.glob(files_pattern)
    for f in files_to_remove:
        os.remove(f)


def main():
    plot_metrics_models()
    plot_metrics_seen()
    plot_execution_time()
    plot_MOSNet_score_models()
    plot_MOSNet_score_seen()

    # remove_mosnet_results()


if __name__ == '__main__':
    main()
