from preprocess import NamedEntity


class Tuple:
    def __init__(self, e1: NamedEntity, e2: NamedEntity, before: str, between: str, after: str, config):
        self.e1 = e1
        self.e2 = e2
        self.bef_tag = before
        self.bet_tag = between
        self.aft_tag = after
