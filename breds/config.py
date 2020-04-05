"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   config.py
"""

import fileinput
import pickle

from breds.seed import Seed


class Config:
    def __init__(self, config_file, positive_seeds, negative_seeds, similarity, confidence):
        self.threshold_similarity = similarity
        self.instance_confidence = confidence

        self.positive_seed_tuples = set()
        self.negative_seed_tuples = set()
        self.e1_type = None
        self.e2_type = None

        self.word2idx = None
        self.emb_matrix = None
        self.vec_dim = 0

        # Loading Config_File
        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.wUnk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.wNeg = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1].strip()

            if line.startswith("word2idx_path"):
                self.word2idx_path = line.split("=")[1].strip()

            if line.startswith("emb_matrix_path"):
                self.emb_matrix_path = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

            if line.startswith("tags_type"):
                self.tag_type = line.split("=")[1].strip()

        assert self.alpha + self.beta + self.gamma == 1

        self.read_seeds(positive_seeds, self.positive_seed_tuples)
        self.read_seeds(negative_seeds, self.negative_seed_tuples)

        fileinput.close()

    def load_word2idx_embmatrix(self):
        print("Loading word2idx and emb_matrix ...\n")
        self.word2idx = pickle.load(open(self.word2idx_path, "rb"))
        self.emb_matrix = pickle.load(open(self.emb_matrix_path, "rb"))
        self.vec_dim = len(self.emb_matrix[0])
        print(self.vec_dim, "dimensions")

    def read_seeds(self, seeds_file: str, holder: set):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                holder.add(seed)
