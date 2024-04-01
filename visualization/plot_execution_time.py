import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_results(root_directory: str):
    results = []
    for root_dir, dirs, files in os.walk(root_directory):
        if 'execution_time.txt' in files:
            if 'stargan' in root_dir:
                model = 'stargan'
            elif 'triann' in root_dir:
                model = 'triann'
            else:
                raise 'No model in path!'
            number_of_speakers = int(re.search(r'\d+', re.search(r'\d+spks', root_dir).group()).group())
            with open(os.path.join(root_dir, 'execution_time.txt')) as f:
                execution_time = json.load(f)
            results.append(
                (model, number_of_speakers, execution_time)
            )
    df = pd.DataFrame(results, columns=['model', 'number_of_speakers', 'execution_time'])
    return df


def filter_results_and_reformat_it(
        df: pd.DataFrame,
        plot_title: str
):
    df_stargan = df[df.model == 'stargan']
    df_triann = df[df.model == 'triann']

    x_ticks = df_stargan['number_of_speakers'].tolist()
    y_label = 'Training time'

    df_stargan = df_stargan.sort_values(['execution_time'])
    df_triann = df_triann.sort_values(['execution_time'])

    times_stargan = df_stargan['execution_time'].tolist()
    times_triann = df_triann['execution_time'].tolist()

    models_plot_properties = [
        ('StarGANv2-VC', 'red', times_stargan),
        ('TriANN-VC', 'blue', times_triann)
    ]

    return plot_title, y_label, x_ticks, models_plot_properties


def bar_plot_of_execution_time(
        plot_title: str,
        y_label: str,
        x_ticks,
        models_plot_properties
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
    # plt.xticks(rotation=90, fontsize=10)

    plt.show()
