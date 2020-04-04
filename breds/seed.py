from preprocess import NamedEntity
import re


class Seed:
    def __init__(self, e1: NamedEntity, e2: NamedEntity):
        self.e1 = e1
        self.e2 = e2

    def __eq__(self, other):
        return self.e1.category == other.e1.category and \
               self.e2.category == other.e2.category and \
               re.sub("[\n ]", "", self.e1.name) == re.sub("[\n ]", "", other.e1.name) and \
               re.sub("[\n ]", "", self.e2.name) == re.sub("[\n ]", "", other.e2.name)