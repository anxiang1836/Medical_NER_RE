"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   run_generate.py
"""

from evaluate import prepare, predict, decode_result
from utils import scan_files, load_file, logger_init
from typing import List
import numpy as np
import argparse
import pickle

logger = logger_init()


def generate_ner(args) -> None:
    """
    总共分成2个步骤：
        Step1 ： 用模型进行实体识别
        Step2 : 对每篇文章按照中文句点进行分割
    Args:
        args:
            --file_root ： root path of data
    """

    file_names = scan_files(args.file_root)  # type:List[str]
    for file in file_names:
        data = load_file(args.file_root, file, "txt")

        # Part1 : 计算得到当前文章的实体识别结果
        prepare_data = prepare(data)  # type:np.ndarray
        result = predict(prepare_data)  # type:np.ndarray
        _, ner_result = decode_result(result=result, sent_pre=prepare_data, sent=data)

        pickle.dump(ner_result, open(args.file_root + file + "_ner.pkl", 'wb'))

        # Part2 ： 将当前文章按照（句号/问号/感叹号）作为划分，并记录到dict中
        start, end = 0, 0
        sentence_split_result = []
        stop_tokens = ["。", "！", "？"]
        for idx, c in enumerate(data):
            if c in stop_tokens:
                end = idx
                sentence_split_result.append((start, end))
                start = end + 1

        pickle.dump(sentence_split_result, open(args.file_root + file + "_sentence_split.pkl", 'wb'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_root", type=str, default="./data/round2/", required=True,
                        help="root path of data")

    args = parser.parse_args()
    generate_ner(args)


if __name__ == "__main__":
    main()
