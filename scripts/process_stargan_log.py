from typing import List, Tuple, Dict
import pandas as pd


def read_lines(path: str) -> List[str]:
    nohup_file = open(path, "r")
    return nohup_file.readlines()


def filter_lines_and_split_train_eval(lines: List[str]) -> Tuple[List[str], List[str]]:
    lines = list(filter(lambda l: not l.startswith("--"), lines))
    lines = list(filter(lambda l: not l.startswith("{"), lines))
    lines = list(filter(lambda l: "time:" not in l, lines))

    train_lines = list(filter(lambda l: l.startswith("train"), lines))
    eval_lines = list(filter(lambda l: l.startswith("eval"), lines))
    # print(len(train_lines))
    # print(len(eval_lines))
    return train_lines, eval_lines


def __process_single_cell_str_into_dict(cell_str: str) -> Dict[str, float]:
    s = cell_str.split(":")
    key, value = s[0].strip(), float(s[1].strip())
    return {key: value}


def __process_single_line_into_dict(line: str) -> Dict[str, float]:
    cells = line.split('|')
    cells = list(map(lambda c: c.strip(), cells))
    key_value_tuples = list(map(lambda c: __process_single_cell_str_into_dict(c), cells))

    single_line_dict = {}
    for d in key_value_tuples:
        single_line_dict.update(d)

    return single_line_dict


def process_lines_into_df(lines: List[str]):
    without_prefix = list(map(lambda l: l[7:], lines))  # ugly i know
    dicts = list(map(lambda c: __process_single_line_into_dict(c), without_prefix))
    print(len(dicts))

    return pd.DataFrame.from_records(dicts)


def main():
    nohup_path = "../StarGANv2-VC/Models/2024-01-05-demo-data-20-speakers/nohup.out"
    lines = read_lines(nohup_path)
    train_lines, eval_lines = filter_lines_and_split_train_eval(lines)

    train_df = process_lines_into_df(train_lines)
    eval_df = process_lines_into_df(eval_lines)

    print(eval_df)


if __name__ == '__main__':
    main()
