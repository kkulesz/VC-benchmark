from typing import List, Tuple, Dict
import pandas as pd
import re


def read_lines(path: str) -> List[str]:
    nohup_file = open(path, "r")
    return nohup_file.readlines()


def __process_single_line(line: str) -> (int, float, float):
    epoch_part, training_part, valid_part = line.split('|')

    epoch = int(re.search(r'\d+', epoch_part).group())
    train_loss = float(re.search(r'\d+\.\d+', training_part).group())
    valid_loss = float(re.search(r'\d+\.\d+', valid_part).group())

    return epoch, train_loss, valid_loss


def filter_lines_and_split_train_eval_into_dfs(lines: List[str]):
    lines = list(filter(lambda l: l.startswith("epoch:"), lines))
    rows = list(map(lambda l: __process_single_line(l), lines))

    df = pd.DataFrame(rows, columns=['epoch', 'train_loss', 'valid_loss'])
    train_df = df[['epoch', 'train_loss']]
    valid_df = df[['epoch', 'valid_loss']]
    return train_df, valid_df


def main():
    nohup_path = "../../Models/triann/demodata/nohup.out"
    lines = read_lines(nohup_path)
    train_df, valid_df = filter_lines_and_split_train_eval_into_dfs(lines)
    print(train_df)
    print(valid_df)


if __name__ == '__main__':
    main()
