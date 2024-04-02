import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import statistics

def read_results(root_directory: str):
    def merge_male_and_female_results(dataframe: pd.DataFrame):
        models = dataframe['model'].unique()
        speakers = dataframe['number_of_speakers'].unique()
        seens = dataframe['seen'].unique()
        results_merged = []
        for m, spks, s in itertools.product(models, speakers, seens):
            filtered = dataframe.loc[
                (dataframe['model'] == m) &
                (dataframe['number_of_speakers'] == spks) &
                (dataframe['seen'] == s)
            ]

            assert len(filtered) == 2
            mcd = statistics.mean(filtered['MCD'])
            snr = statistics.mean(filtered['SNR'])
            merged = (m, spks, s, mcd, snr)

            results_merged.append(merged)
        final = pd.DataFrame(results_merged, columns=['model', 'number_of_speakers', 'seen', 'MCD', 'SNR'])
        return final

    results = []
    for root_dir, dirs, files in os.walk(root_directory):
        if 'results.json' in files:

            if 'stargan' in root_dir:
                model = 'stargan'
            elif 'triann' in root_dir:
                model = 'triann'
            else:
                raise 'No model in path!'
            number_of_speakers = int(re.search(r'\d+', re.search(r'\d+spks', root_dir).group()).group())
            seen = 'unseen' not in root_dir
            directory = os.path.basename(root_dir)
            male_to_female = directory in ['VCC2TM2_in_voice_of_VCC2TF2', 'VCC2SM1_in_voice_of_VCC2SF1']
            with open(os.path.join(root_dir, 'results.json')) as f:
                metric_results = json.load(f)

            results.append(
                (model, number_of_speakers, seen, directory, male_to_female,
                 metric_results['MCD'],
                 metric_results['SNR'])
            )
    df = pd.DataFrame(results,
                      columns=['model', 'number_of_speakers', 'seen', 'directory', 'male_to_female', 'MCD', 'SNR'])

    df = merge_male_and_female_results(df)

    return df


def filter_results_and_reformat_it_between_models(
        seen: bool,
        metric: str,
        df: pd.DataFrame,
        plot_title: str
):
    df = df[df.seen == seen]

    df_stargan = df[df.model == 'stargan']
    df_triann = df[df.model == 'triann']

    df_stargan = df_stargan.sort_values(['number_of_speakers'])
    df_triann = df_triann.sort_values(['number_of_speakers'])

    metric_stargan = df_stargan[metric].tolist()
    metric_triann = df_triann[metric].tolist()

    min_value = min(metric_triann + metric_stargan)
    max_value = max(metric_triann + metric_stargan)
    if metric == 'MCD':
        y_limits = [5, 8]
    elif metric == 'SNR':
        y_limits = [0, 2.7]
    else:
        y_limits = (0.8 * min_value, 1.05 * max_value)

    x_ticks = df_stargan['number_of_speakers'].tolist()
    y_label = metric
    models_plot_properties = [
        ('StarGANv2-VC', 'red', metric_stargan),
        ('TriANN-VC', 'blue', metric_triann)
    ]

    return plot_title, y_label, x_ticks, models_plot_properties, y_limits

def filter_results_and_reformat_it_between_seen_and_unseen_speakers(
        model: str,
        metric: str,
        df: pd.DataFrame,
        plot_title: str
):
    df = df[df.model == model]

    df_seen = df[df.seen == True]
    df_unseen = df[df.seen == False]

    df_seen = df_seen.sort_values(['number_of_speakers'])
    df_unseen = df_unseen.sort_values(['number_of_speakers'])

    metric_seen = df_seen[metric].tolist()
    metric_unseen = df_unseen[metric].tolist()

    min_value = min(metric_unseen + metric_seen)
    max_value = max(metric_unseen + metric_seen)
    if metric == 'MCD':
        y_limits = [5, 8]
    elif metric == 'SNR':
        y_limits = [0, 2.7]
    else:
        y_limits = (0.8 * min_value, 1.05 * max_value)


    x_ticks = df_seen['number_of_speakers'].tolist()
    y_label = metric
    models_plot_properties = [
        ('Seen', 'green', metric_seen),
        ('Unseen', 'purple', metric_unseen)
    ]

    return plot_title, y_label, x_ticks, models_plot_properties, y_limits

def bar_plot_of_metric_over_number_of_speakers_for_each_model(
        plot_title: str,
        y_label: str,
        x_ticks,
        models_plot_properties,
        y_limits
):
    bar_width = 0.25
    fig = plt.subplots(figsize=(12, 8))

    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.xticks([r + (bar_width / 2) * (len(models_plot_properties) - 1) for r in range(len(x_ticks))], x_ticks)
    for idx, (label, color, values) in enumerate(models_plot_properties):
        br = [x + bar_width * idx for x in np.arange(len(values))]
        plt.bar(br, values, color=color, width=bar_width, edgecolor='grey', label=label)
    plt.legend()
    plt.ylim(ymin=y_limits[0], ymax=y_limits[1])
    # plt.xticks(rotation=90, fontsize=10)

    plt.show()