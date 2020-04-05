"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   w2c_trainer.py
"""

from utils import scan_files, load_file

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import multiprocessing


# 字-Vec的训练器
class Char2VecTrainer:
    def __init__(self, root: str, w2v_file_path: str):
        self.root = root
        self._w2v_file_path = w2v_file_path

    def prepare_data(self):
        with open(self._w2v_file_path, 'w', encoding='utf-8') as f:
            file_names = scan_files(self.root)
            for name in file_names:
                data = load_file(self.root, name, "txt")
                data = " ".join(data)
                f.write(data)
                f.write("\n")
        return

    def train(self, output: str, window: int, sg: int, hs: int, negative: int, emb_size=128):
        model = Word2Vec(LineSentence(self._w2v_file_path), size=emb_size, window=window, sg=sg, hs=hs,
                         negative=negative, workers=multiprocessing.cpu_count())
        model.save(output)
        return

    def load(self):
        w2v_model = Word2Vec.load(self._w2v_file_path)
        print('w2v的模型维度是：{}'.format(w2v_model.wv.vector_size))
        print('w2v的模型的词表总长是：{}'.format(len(w2v_model.wv.vocab)))
        return w2v_model
