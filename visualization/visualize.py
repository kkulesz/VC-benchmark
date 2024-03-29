import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_results(root_directory: str):
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
    return df


def filter_results_and_reformat_it(
        seen: bool,
        male_to_female: bool,
        metric: str,
        df: pd.DataFrame,
        plot_title: str
):
    df = df[df.seen == seen]
    df = df[df.male_to_female == male_to_female]

    df_stargan = df[df.model == 'stargan']
    df_triann = df[df.model == 'triann']

    df_stargan = df_stargan.sort_values(['number_of_speakers'])
    df_triann = df_triann.sort_values(['number_of_speakers'])

    metric_stargan = df_stargan[metric].tolist()
    metric_triann = df_triann[metric].tolist()

    x_ticks = df_stargan['number_of_speakers'].tolist()
    y_label = metric
    models_plot_properties = [
        ('StarGANv2-VC', 'red', metric_stargan + [800]), # TODO remove those
        ('TriANN-VC', 'blue', metric_triann + [900]) # TODO: remove those
    ]

    return plot_title, y_label, x_ticks, models_plot_properties


def bar_plot_of_metric_over_number_of_speakers_for_each_model(
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


def main():
    # plot_title = 'MCD over number os speakers in training dataset'
    # y_label = 'MCD'
    # models_plot_properties = [
    #     ('StarGANv2-VC', 'red', [5, 15, 30, 18, 21]),
    #     ('TriANN-VC', 'blue', [6, 18, 28, 21, 37]),
    #     ('TEST', 'green', [2, 3, 4, 5, 6])
    # ]
    # x_ticks = ['2', '5', '10', '25', '50']

    df = read_results('../../samples')
    plot_title, y_label, x_ticks, models_plot_properties = \
        filter_results_and_reformat_it(seen=True, male_to_female=True, metric='MCD', df=df, plot_title='MCD over number os speakers in training dataset')

    bar_plot_of_metric_over_number_of_speakers_for_each_model(plot_title, y_label, x_ticks, models_plot_properties)


if __name__ == '__main__':
    main()
