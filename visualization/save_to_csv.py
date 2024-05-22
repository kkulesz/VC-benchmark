import pandas as pd
import os

RESULTS_DIR = '../../samples'


def _split_into_model_and_seen(results_df):
    stargan_seen = results_df.loc[(results_df.model == 'stargan') & (results_df.seen == True)]
    stargan_unseen = results_df.loc[(results_df.model == 'stargan') & (results_df.seen == False)]

    triann_seen = results_df.loc[(results_df.model == 'triann') & (results_df.seen == True)]
    triann_unseen = results_df.loc[(results_df.model == 'triann') & (results_df.seen == False)]

    return stargan_seen, stargan_unseen, triann_seen, triann_unseen


def _build_final_csv_and_save(stargan_seen_df, stargan_unseen_df, triann_seen_df, triann_unseen_df, metric_col,
                              filename):
    assert len(stargan_seen_df.index) == len(stargan_unseen_df.index) == len(triann_seen_df.index) == len(
        triann_unseen_df.index)

    number_of_spks = stargan_seen_df['number_of_speakers'].tolist()
    stargan_seen = [f'{elem:.2f}' for elem in stargan_seen_df[metric_col].tolist()]
    stargan_unseen = [f'{elem:.2f}' for elem in stargan_unseen_df[metric_col].tolist()]
    triann_seen = [f'{elem:.2f}' for elem in triann_seen_df[metric_col].tolist()]
    triann_unseen = [f'{elem:.2f}' for elem in triann_unseen_df[metric_col].tolist()]

    dict = {
        'Liczba mówców': number_of_spks,
        'StarGANv2-VC-widziani': stargan_seen,
        'TriANN-VC-widziani': triann_seen,
        'StarGANv2-VC-niewidziani': stargan_unseen,
        'TriANN-VC-niewidziani': triann_unseen
    }

    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(os.path.join('../../Tables', filename)), index=False)


def _save_metrics():
    import plot_metrics

    results_df = plot_metrics.read_results(RESULTS_DIR)
    results_df = results_df.sort_values(['number_of_speakers'])
    eng_stargan_seen, eng_stargan_unseen, eng_triann_seen, eng_triann_unseen = _split_into_model_and_seen(results_df)

    _build_final_csv_and_save(eng_stargan_seen, eng_stargan_unseen, eng_triann_seen, eng_triann_unseen, 'MCD',
                              'MCD.csv')
    _build_final_csv_and_save(eng_stargan_seen, eng_stargan_unseen, eng_triann_seen, eng_triann_unseen, 'SNR',
                              'SNR.csv')


def _save_mosnet():
    import plot_mosnet

    results_df = plot_mosnet.read_results(RESULTS_DIR)
    results_df = results_df.drop('directory', axis=1)
    results_df = results_df.sort_values(['number_of_speakers'])

    english = results_df[results_df.language == 'english']
    eng_stargan_seen, eng_stargan_unseen, eng_triann_seen, eng_triann_unseen = _split_into_model_and_seen(english)
    _build_final_csv_and_save(eng_stargan_seen, eng_stargan_unseen, eng_triann_seen, eng_triann_unseen, 'mosnet_score',
                              'eng_mosnet.csv')

    polish = results_df[results_df.language == 'polish']
    pol_stargan_seen, pol_stargan_unseen, pol_triann_seen, pol_triann_unseen = _split_into_model_and_seen(polish)
    _build_final_csv_and_save(pol_stargan_seen, pol_stargan_unseen, pol_triann_seen, pol_triann_unseen, 'mosnet_score',
                              'pol_mosnet.csv')


def save_results_to_csv():
    _save_metrics()
    _save_mosnet()


def main():
    save_results_to_csv()


if __name__ == '__main__':
    main()
