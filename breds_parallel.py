"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   breds_parallel.py
@Modify Time :   2020/4/5 0005 14:39 
"""

import queue
import pickle
import argparse
import multiprocessing
from collections import defaultdict
from typing import List

from utils import scan_files, load_file

from breds.config import Config
from breds.tuple import Tuple


class BREDS:
    def __init__(self, args):
        if args.num_cores == 0:
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = args.num_cores
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)  # 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认空list
        self.config = Config(args.config_file, args.positive_seeds_file, args.negative_seeds_file, args.similarity,
                             args.confidence)

    def generate_tuples(self, data_dir: str):
        """
        用于从源数据中，用多线程的方式生成tuples
        Args:
            data_dir: 数据存储的路径，其中包括：
                      eg. 源文章名称    __ data/round2/0.txt
                          NER结果名称   __ data/round2/0_ner.pkl
                          文章分句结果   __ data/round2/0_sentence_split.pkl
        """

        # Step1 : load word2idx and emb_matrix
        self.config.load_word2idx_embmatrix()

        # Step2 : 生成候选关系对
        instances = list()
        file_names = scan_files(data_dir)

        for file in file_names:
            passage = load_file(data_dir, file, "txt")  # type:str
            sent_split = pickle.load(open(data_dir + file + "_sentence_split.pkl", "rb"))  # type:List[tuple]
            ner_result = pickle.load(open(data_dir + file + "_ner.pkl", "rb"))  # type:List[tuple]

            sent_split.sort(key=lambda x: x[0])

            # Step2.1 : 找出属于e1与e2的实体
            e1_entities, e2_entities = list(), list()
            for e in ner_result:
                # e是个4元组，例如：('Disease', 1, 10, '糖尿病下肢动脉病变')
                if e[0] == self.config.e1_type:
                    e1_entities.append(e)
                elif e[0] == self.config.e2_type:
                    e2_entities.append(e)
            e1_entities.sort(key=lambda x: x[1])
            e2_entities.sort(key=lambda x: x[1])

            # Step2.2 : 对每一个e1去找到候选的e2，并确定三元组<BEF，BET，AFT,sequence_tag>
            for e1 in e1_entities:
                e1_start, e1_end = e1[1], e1[2]
                cur_sentence_idx = -1
                for idx, s in enumerate(sent_split):
                    if s[0] <= e1_start and s[1] >= e1_end:
                        cur_sentence_idx = idx
                        break
                # 根据当前实体的位置确定了寻找e2的上下界：即 上一句 + 当前句 + 下一句
                search_e2_start = sent_split[cur_sentence_idx - 1 if cur_sentence_idx > 1 else 0][0]
                search_e2_end = sent_split[cur_sentence_idx + 1 if cur_sentence_idx < len(sent_split) - 1 \
                    else len(sent_split) - 1][1]

                for i in range(len(e2_entities)):
                    e2 = e2_entities[i]
                    e2_start = e2[1]
                    e2_end = e2[2]
                    if e2_end < search_e2_start:
                        continue
                    elif e2_start > search_e2_end:
                        break
                    elif e2_start >= search_e2_start and e2_end <= search_e2_end:
                        if e1_end == e2_start:
                            # 情况(1)：e1在e2前，且紧挨着
                            before = passage[search_e2_start:e1_start]
                            between = ""
                            after = passage[e2_end:search_e2_end]
                            t = Tuple(e1[3], e2[3], sequence_tag=True, before=before, between=between, after=after,
                                      config=self.config)
                            instances.append(t)
                        elif e2_end == e1_start:
                            # 情况（2）：e1在e2后，且紧挨着
                            before = passage[search_e2_start:e2_start]
                            between = ""
                            after = passage[e1_end:search_e2_end]
                            t = Tuple(e1[3], e2[3], sequence_tag=False, before=before, between=between, after=after,
                                      config=self.config)
                            instances.append(t)
                        elif e1_end < e2_start:
                            # 情况（3）：e1在e2前，不挨着
                            before = passage[search_e2_start:e1_start]
                            between = passage[e1_end:e2_start]
                            after = passage[e2_end:search_e2_end]
                            t = Tuple(e1[3], e2[3], sequence_tag=True, before=before, between=between, after=after,
                                      config=self.config)
                            instances.append(t)
                        elif e2_end < e1_start:
                            # 情况（4）：e1在e2后，不挨着
                            before = passage[search_e2_start:e2_start]
                            between = passage[e2_end:e1_start]
                            after = passage[e1_end:search_e2_end]
                            t = Tuple(e1[3], e2[3], sequence_tag=False, before=before, between=between, after=after,
                                      config=self.config)
                            instances.append(t)

        # Stpe3 : 持久化
        pickle.dump(instances, open("./saved_model_files/RE_candidate_instances.pkl", "wb"))

    def similarity_3_contexts(self, t: Tuple, p: Tuple) -> float:
        bef, bet, aft = 0, 0, 0
        # TODO 貌似应该增加参数的
        return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="./parameters.cfg", required=True,
                        help="config files for RE")
    parser.add_argument("--positive_seeds_file", type=str, default="./breds/seeds_positive.txt", required=True,
                        help="positive seeds files for RE")
    parser.add_argument("--negative_seeds_file", type=str, default="./breds/seeds_negative.txt", required=True,
                        help="negative seeds files for RE")
    parser.add_argument("--similarity", type=float, default=0.6, required=True,
                        help="similarity score between instance and pattern")
    parser.add_argument("--confidence", type=float, default=0.6, required=True,
                        help="confidence of instance to be a new seed")
    parser.add_argument("--num_cores", type=int, default=0, help="cpu_count for multiprocessing")

    args = parser.parse_args()

    breds = BREDS(args)
