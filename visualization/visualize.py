import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MCD = 'MCD'
SNR = 'SNR'


def plot_metrics():
    import plot_metrics
    seen = True
    male_to_female = True
    metric = SNR

    df = plot_metrics.read_results('../../samples')
    plot_title, y_label, x_ticks, models_plot_properties, bottom = \
        plot_metrics.filter_results_and_reformat_it(seen=seen, male_to_female=male_to_female, metric=metric, df=df,
                                                    plot_title=f'{metric} over number of {"seen" if seen else "unseen"} speakers in training dataset')

    plot_metrics.bar_plot_of_metric_over_number_of_speakers_for_each_model(plot_title, y_label, x_ticks,
                                                                           models_plot_properties, bottom)


def plot_execution_time():
    import plot_execution_time
    df = plot_execution_time.read_results('../../Models')
    print(df)
    plot_title, y_label, x_ticks, models_plot_properties = \
        plot_execution_time.filter_results_and_reformat_it(df=df, plot_title='Training time')
    plot_execution_time.bar_plot_of_execution_time(plot_title, y_label, x_ticks, models_plot_properties)


def main():
    plot_metrics()
    # plot_execution_time()


if __name__ == '__main__':
    main()
