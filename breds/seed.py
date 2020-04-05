"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   seed.py
"""
import re


class Seed:
    def __init__(self, e1: str, e2: str):
        self.e1 = e1
        self.e2 = e2

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2)

    def __eq__(self, other):
        # 主要是实体e中，会有空格和\n将一个实体给分隔开
        return re.sub("[\n ]", "", self.e1) == re.sub("[\n ]", "", other.e1) and \
               re.sub("[\n ]", "", self.e2) == re.sub("[\n ]", "", other.e2)
