import matplotlib.pyplot as plt
import numpy as np


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
    plot_title = 'MCD over number os speakers in training dataset'
    y_label = 'MCD'
    models_plot_properties = [
        ('StarGANv2-VC', 'red', [5, 15, 30, 18, 21]),
        ('TriANN-VC', 'blue', [6, 18, 28, 21, 37]),
        ('TEST', 'green', [2, 3, 4, 5, 6])
    ]
    x_ticks = ['2', '5', '10', '25', '50']

    bar_plot_of_metric_over_number_of_speakers_for_each_model(plot_title, y_label, x_ticks, models_plot_properties)


if __name__ == '__main__':
    main()
