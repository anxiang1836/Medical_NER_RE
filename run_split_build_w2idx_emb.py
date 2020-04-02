from preprocess import DataSet
from utils import scan_files, logger_init

from sklearn.model_selection import ShuffleSplit
from gensim.models import Word2Vec
import argparse
import pickle
import numpy as np

logger = logger_init()


def split_build(args):
    file_paths = scan_files(args.root_path)
    logger.info("***** Running Start *****")
    logger.info("Step1 : Begin to build char2idx")
    # Step1 ： 建立整个数据集上的word2idx
    whole_set = DataSet(args.root_path, file_paths, vocab_size=-1)
    char2idx = whole_set.char2idx
    pickle.dump(char2idx, open(args.word2idx_output_path, 'wb'))
    del whole_set
    logger.info("Step1 : Done!word2idx_output_path is %s", args.word2idx_output_path)

    # Step2 ： 切分训练集/测试集/验证集
    logger.info("Step2 : Begin to Split DatSet")
    rs = ShuffleSplit(n_splits=1, test_size=.15, random_state=2019)
    train_idx, test_idx = next(rs.split(file_paths))
    train_file_names = [file_paths[idx] for idx in train_idx]
    test_file_names = [file_paths[idx] for idx in test_idx]

    rs = ShuffleSplit(n_splits=1, test_size=.20, random_state=2019)
    train_idx, val_idx = next(rs.split(train_file_names))
    train_file_names = [file_paths[idx] for idx in train_idx]
    val_file_names = [file_paths[idx] for idx in val_idx]
    logger.info("Step2 : Done!")

    # Step3 : 数据集转化为idx并持久化
    logger.info("Step3 : Begin to save DatSet")
    trainset = DataSet(args.root_path, train_file_names, char2idx)
    valset = DataSet(args.root_path, val_file_names, char2idx)
    testset = DataSet(args.root_path, test_file_names, char2idx)

    pickle.dump(trainset, open('./saved_model_files/trainset.pkl', 'wb'))
    pickle.dump(valset, open('./saved_model_files/valset.pkl', 'wb'))
    pickle.dump(testset, open('./saved_model_files/testset.pkl', 'wb'))
    logger.info("Step3 : Done!")

    # Step4 ： 根据word2idx创建emb_matrix
    logger.info("Step4 : Begin to build emb_matrix")
    char2vec_model = Word2Vec.load(args.w2v_output_path)
    vec_size = char2vec_model.wv.vector_size
    emb_matrix = np.zeros(vec_size)

    def random_vec(size: int) -> np.array:
        # 创建一个随机向量，用于赋值unk
        vec = np.random.random(size=size)
        vec = vec - vec.mean()
        return vec

    for c in char2idx.keys():
        if c is "_padding":
            char2idx[c] = 0
        elif c is "_unk":
            emb = random_vec(vec_size)
            emb_matrix = np.vstack((emb_matrix, emb))
            char2idx[c] = 1
        else:
            if c in [" ", "\n"]:
                idx = emb_matrix.shape[0]
                emb = random_vec(vec_size)
                emb_matrix = np.vstack((emb_matrix, emb))
                char2idx[c] = idx
            elif c not in char2vec_model.wv.vocab.keys():
                idx = char2idx["_unk"]
                char2idx[c] = idx
            else:
                idx = emb_matrix.shape[0]
                emb = char2vec_model.wv[c]
                emb_matrix = np.vstack((emb_matrix, emb))
                char2idx[c] = idx
    pickle.dump(emb_matrix, open(args.emb_matrix_output_path, 'wb'))
    logger.info("Step4 : Done!Emb_matrix_output_path is %s", args.emb_matrix_output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default="./data/round1/train/", required=True,
                        help="root path of data")
    parser.add_argument("--test_size", type=float, default=.15, required=True, help="ratio of test samples")
    parser.add_argument("--val_size", type=float, default=.2, required=True, help="ratio of val samples")

    parser.add_argument("--word2idx_output_path", type=str, default="./saved_model_files/word2idx.pkl",
                        required=True, help="output path of word2idx")
    parser.add_argument("--emb_matrix_output_path", type=str, default="./saved_model_files/emb_matrix.pkl",
                        required=True, help="output path of word2idx")
    parser.add_argument("--w2v_output_path", type=str, default="./saved_model_files/char2vec.model",
                        required=True, help="output path of w2v")

    args = parser.parse_args()

    split_build(args)


if __name__ == "__main__":
    main()
