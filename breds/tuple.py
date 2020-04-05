"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   tuple.py
"""

import re
import numpy as np
from breds.config import Config
from typing import List


class Tuple:
    def __init__(self, e1: str, e2: str, sequence_tag: bool, before: str, between: str, after: str, config):
        """
        Args:
            sequence_tag: 主要是考虑到先后顺序会导致有不同的模式，那么在对模式做cluster时，按照顺序不同区别来做
        """
        self.e1 = e1
        self.e2 = e2
        self.sequence_tag = sequence_tag  # 用于标记e1与e2的顺序关系，如果e1在e2之前，True；否则，False
        self.bef_tags = before
        self.bet_tags = between
        self.aft_tags = after
        self.bef_vector = None
        self.bet_vector = None
        self.aft_vector = None
        self.__construct_vectors__(config)

    def __eq__(self, other) -> bool:
        return re.sub("[\n ]", "", self.e1) == re.sub("[\n ]", "", other.e1) and \
               re.sub("[\n ]", "", self.e2) == re.sub("[\n ]", "", other.e2) and \
               self.bef_tags == other.bef_tags and \
               self.bet_tags == other.bet_tags and \
               self.aft_tags == other.aft_tags

    def __str__(self) -> str:
        return str(self.e1 + '\t' + self.e2 + '\t' + self.bef_tags + '\t' +
                   self.bet_tags + '\t' + self.aft_tags)

    def __hash__(self) -> int:
        return hash(self.e1) ^ hash(self.e2) ^ hash(self.bef_tags) ^ \
               hash(self.bet_tags) ^ hash(self.aft_tags)

    def __construct_vectors__(self, config):
        # 是否需要对pattern的无关词/进行清洗：如果需要的话，添加到config中
        self.bef_vector = self.__pattern2vector_sum__([c for c in self.bef_tags], config)
        self.bet_vector = self.__pattern2vector_sum__([c for c in self.bet_tags], config)
        self.aft_vector = self.__pattern2vector_sum__([c for c in self.aft_tags], config)

    @staticmethod
    def __pattern2vector_sum__(tokens: List[str], config: Config) -> np.ndarray:
        """
        计算pattern的加和向量，如果tokens的长度为0，例如e1与e2紧挨着的话，那么返回零向量
        Args:
            tokens: 提取的模式的字符串List
            config: 配置文件，用于读取word2idx与emb_matrix

        Returns:
            pattern_vector:返回pattern对应的和向量
        """
        pattern_vector = np.zeros(config.vec_dim)
        if len(tokens) > 0:
            for t in tokens:
                if t in config.word2idx.keys():
                    vector = config.emb_matrix[config.word2idx[t]]
                else:
                    vector = config.emb_matrix[config.word2idx["_unk"]]
                pattern_vector += vector
        elif len(tokens) == 0:
            # 如果pattern中的字数为0的话，那就返回一个零向量
            pass
        return pattern_vector
