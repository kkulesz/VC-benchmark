import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

MCD = 'MCD'
SNR = 'SNR'


def plot_metrics():
    import plot_metrics
    seen = True
    metric = MCD

    df = plot_metrics.read_results('../../samples')
    plot_title, y_label, x_ticks, models_plot_properties, bottom = \
        plot_metrics.filter_results_and_reformat_it(seen=seen, metric=metric, df=df,
                                                    plot_title=f'{metric} over number of {"seen" if seen else "unseen"} speakers in training dataset')

    plot_metrics.bar_plot_of_metric_over_number_of_speakers_for_each_model(
        plot_title, y_label, x_ticks, models_plot_properties, bottom)


def plot_execution_time():
    import plot_execution_time
    df = plot_execution_time.read_results('../../Models')
    print(df)
    plot_title, y_label, x_ticks, models_plot_properties = \
        plot_execution_time.filter_results_and_reformat_it(df=df, plot_title='Training time')
    plot_execution_time.bar_plot_of_execution_time(plot_title, y_label, x_ticks, models_plot_properties)


def plot_mosnet_score():
    import plot_mosnet
    seen = True

    df = plot_mosnet.read_results('../../samples')
    print(df)
    plot_title, y_label, x_ticks, models_plot_properties, bottom = \
        plot_mosnet.filter_results_and_reformat_it(seen=seen, df=df,
                                                   plot_title=f'MOSNet score over number of {"seen" if seen else "unseen"} speakers in training dataset')

    plot_mosnet.bar_plot_of_MOSNet_score_over_number_of_speakers(
        plot_title, y_label, x_ticks, models_plot_properties, bottom)


def remove_mosnet_results(directory: str = '../../samples'):
    files_pattern = os.path.join(directory, '*/*/*/*/MOSnet_result_raw.txt')
    files_to_remove = glob.glob(files_pattern)
    for f in files_to_remove:
        os.remove(f)


def main():
    plot_metrics()
    # plot_execution_time()
    # plot_mosnet_score()

    # remove_mosnet_results()


if __name__ == '__main__':
    main()
