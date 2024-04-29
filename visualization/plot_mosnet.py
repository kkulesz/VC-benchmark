import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Y_LIMITS = (2.8, 4.85)


def read_results(root_directory: str):
    results = []
    for root_dir, dirs, files in os.walk(root_directory):
        if 'MOSnet_result_raw.txt' in files:

            if 'stargan' in root_dir:
                model = 'stargan'
            elif 'triann' in root_dir:
                model = 'triann'
            else:
                raise 'No model in path!'
            number_of_speakers = int(re.search(r'\d+', re.search(r'\d+spks', root_dir).group()).group())
            seen = 'unseen' not in root_dir
            directory = os.path.basename(root_dir)
            with open(os.path.join(root_dir, 'MOSnet_result_raw.txt')) as f:
                mosnet_score = float(re.search(r'\d+.\d+', re.search(r'Average: \d+.\d+', f.read()).group()).group())

            results.append(
                (model, number_of_speakers, seen, directory, mosnet_score)
            )
    df = pd.DataFrame(results,
                      columns=['model', 'number_of_speakers', 'seen', 'directory', 'mosnet_score'])
    return df


def filter_results_and_reformat_it_between_models(
        seen: bool,
        df: pd.DataFrame,
        plot_title: str
):
    df = df[df.seen == seen]

    df_stargan = df[df.model == 'stargan']
    df_triann = df[df.model == 'triann']

    df_stargan = df_stargan.sort_values(['number_of_speakers'])
    df_triann = df_triann.sort_values(['number_of_speakers'])

    metric_stargan = df_stargan['mosnet_score'].tolist()
    metric_triann = df_triann['mosnet_score'].tolist()

    x_ticks = df_stargan['number_of_speakers'].tolist()
    models_plot_properties = [
        ('StarGANv2-VC', 'deepskyblue', metric_stargan),
        ('TriANN-VC', 'lightsalmon', metric_triann)
    ]

    return plot_title, x_ticks, models_plot_properties, Y_LIMITS


def filter_results_and_reformat_it_between_seen(
        model: str,
        df: pd.DataFrame,
        plot_title: str,
        bar_labels
):
    df = df[df.model == model]

    df_seen = df[df.seen == True]
    df_unseen = df[df.seen == False]

    df_seen = df_seen.sort_values(['number_of_speakers'])
    df_unseen = df_unseen.sort_values(['number_of_speakers'])

    metric_seen = df_seen['mosnet_score'].tolist()
    metric_unseen = df_unseen['mosnet_score'].tolist()

    x_ticks = df_seen['number_of_speakers'].tolist()
    models_plot_properties = [
        (bar_labels[0], 'red', metric_seen),
        (bar_labels[1], 'blue', metric_unseen)
    ]

    return plot_title, x_ticks, models_plot_properties, Y_LIMITS


def bar_plot_of_MOSNet_score_over_number_of_speakers(
        plot_title: str,
        y_label: str,
        x_label: str,
        x_ticks,
        models_plot_properties,
        y_limits,
        plot_filename: str,
        ground_truth_seen=None,
        ground_truth_unseen=None
):
    bar_width = 0.25
    fig = plt.subplots(figsize=(12, 8))

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.title(plot_title)
    if ground_truth_seen:
        plt.axhline(y=ground_truth_seen[0], color='r', linestyle=':', label=ground_truth_seen[1])
    if ground_truth_unseen:
        plt.axhline(y=ground_truth_unseen[0], color='b', linestyle=':', label=ground_truth_unseen[1])

    plt.xticks([r + (bar_width / 2) * (len(models_plot_properties) - 1) for r in range(len(x_ticks))], x_ticks)
    for idx, (label, color, values) in enumerate(models_plot_properties):
        br = [x + bar_width * idx for x in np.arange(len(values))]
        plt.bar(br, values, color=color, width=bar_width, edgecolor='grey', label=label)
    plt.legend(loc='upper right')
    plt.ylim(ymin=y_limits[0], ymax=y_limits[1])
    # plt.xticks(rotation=90, fontsize=10)

    plt.savefig(plot_filename, dpi=600)

    plt.show()
