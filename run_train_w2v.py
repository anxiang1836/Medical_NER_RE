"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   run_train_w2v.py
"""

import argparse
from model import Char2VecTrainer
from utils import logger_init

logger = logger_init()


def train_w2v(args):
    logger.info("***** Running training w2v *****")
    char2vec = Char2VecTrainer(root=args.root_path, w2v_file_path=args.w2v_file_path)

    char2vec.prepare_data()
    logger.info("Step 1: Prepare_data for w2v is DONE. Files-Saved-Path = %s", args.w2v_file_path)

    char2vec.train(output=args.w2v_output_path, emb_size=args.emb_size, window=args.window_size, sg=args.sg,
                   hs=args.hs, negative=args.negative)
    logger.info("Step 2: Training w2v is DONE. W2V-OUTPUT-Path = %s", args.w2v_output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/round1/train/",
                        required=True, help="origin_data_path")
    parser.add_argument("--w2v_file_path", type=str, default="./saved_model_files/char2vec_prepareData.txt",
                        required=True, help="to be trained files path of w2v")
    parser.add_argument("--w2v_output_path", type=str, default="./saved_model_files/char2vec.model",
                        required=True, help="output path of w2v")

    parser.add_argument("--emb_size", type=int, default=256, required=True, help="embedding size of w2v")
    parser.add_argument("--window_size", type=int, default=5, required=True, help="window size of w2v")
    parser.add_argument("--sg", type=int, default=0, required=False, help="Skip-Gram or not;Default is 0")
    parser.add_argument("--hs", type=int, default=0, required=False, help="Hierarchical-Softmax or not;Default is 0")
    parser.add_argument("--negative", type=int, default=3, required=True, help="Negative-Sampling counts; Default is 3")

    args = parser.parse_args()
    train_w2v(args)


if __name__ == "__main__":
    main()
