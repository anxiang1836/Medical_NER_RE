from evaluate import prepare, predict, decode_result
from utils import scan_files, load_file, logger_init
from typing import List
import numpy as np
import argparse
import pickle

logger = logger_init()


def generate_ner(args) -> None:
    file_names = scan_files(args.file_root)  # type:List[str]
    for file in file_names:
        data = load_file(args.file_root, file, "txt")
        prepare_data = prepare(data)  # type:np.ndarray
        result = predict(prepare_data)  # type:np.ndarray
        _, ner_result = decode_result(result=result, sent_pre=prepare_data, sent=data)

        pickle.dump(ner_result, open(args.file_root + file + "_ner.pkl", 'wb'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_root", type=str, default="./data/round2/", required=True,
                        help="root path of data")

    args = parser.parse_args()
    generate_ner(args)


if __name__ == "__main__":
    main()
