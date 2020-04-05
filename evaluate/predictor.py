"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   predictor.py
"""

import keras
import keras_contrib
import pickle
import numpy as np
from preprocess import CATEGORY
from typing import List

BATCH_SIZE = 64
MAX_LEN = 70
PAD_LEN = 10

word2idx = pickle.load(open('./saved_model_files/word2idx.pkl', 'rb'))
emb_matrix = pickle.load(open('./saved_model_files/emb_matrix.pkl', 'rb'))
model = keras.models.load_model("./saved_model_files/bi_lstm_crf_4_3_2_39.h5",
                                custom_objects={"CRF": keras_contrib.layers.CRF,
                                                "crf_loss": keras_contrib.losses.crf_loss,
                                                "crf_viterbi_accuracy": keras_contrib.metrics.crf_viterbi_accuracy})


def prepare(sent: str) -> np.array:
    # Step1--滑动窗切分句子:
    #   上一个句子的pad_len + 当前窗 + 下一个句子的pad_len
    sent = [c for c in sent]
    if len(sent) // MAX_LEN < 1:
        sent = [PAD_LEN * ["_padding"] + sent + (MAX_LEN - len(sent) + PAD_LEN) * ["_padding"]]
    else:
        cut = []
        for i in range(int(len(sent) // MAX_LEN) + 1):
            cur_sent = sent[i * MAX_LEN: (i + 1) * MAX_LEN]
            if i == 0:
                # 开始
                cur_sent = PAD_LEN * ["_padding"] + cur_sent + sent[(i + 1) * MAX_LEN:(i + 1) * MAX_LEN + PAD_LEN]
            elif i == int(len(sent) // MAX_LEN):
                # 结束
                if len(cur_sent) > 0:
                    cur_sent = sent[i * MAX_LEN - PAD_LEN: i * MAX_LEN] + cur_sent + (
                            MAX_LEN - len(cur_sent) + PAD_LEN) * ["_padding"]
                else:
                    break
            else:
                # 中间
                cur_sent = sent[i * MAX_LEN - PAD_LEN: i * MAX_LEN] + cur_sent + \
                           sent[(i + 1) * MAX_LEN:(i + 1) * MAX_LEN + PAD_LEN]
                if len(cur_sent) < MAX_LEN + 2 * PAD_LEN:
                    cur_sent = cur_sent + (MAX_LEN + 2 * PAD_LEN - len(cur_sent)) * ["_padding"]
            cut.append(cur_sent)
        sent = cut

    # Step2--sent2idx:
    #   上一个句子的pad_len + 当前窗 + 下一个句子的pad_len
    for idx, s in enumerate(sent):
        num_list = []
        for c in s:
            if word2idx.get(c) is not None:
                num_list.append(word2idx.get(str(c)))
            elif word2idx.get(c) is None:
                num_list.append(word2idx.get("_unk"))
        sent[idx] = num_list
    return np.array(sent)


def predict(sent2idx: np.array) -> np.array:
    return model.predict(sent2idx, batch_size=BATCH_SIZE, verbose=True)


def decode_result(result: np.array, sent_pre: np.array, sent: str) -> (List, List):
    # step1: 去除padding
    merge_result = []
    for r, s in zip(result, sent_pre):
        r = r[10:-10].tolist()
        s = s[10:-10].tolist()
        r_not_pad = []
        for idx, each in enumerate(s):
            if each != 0:
                r_not_pad.append(r[idx])
        merge_result += r_not_pad

    # step2：取出最大值
    label_result = []
    for label_vec in merge_result:
        label_idx = np.argmax(label_vec)
        label_result.append(label_idx)
    # return label_result

    # Step3：得到标注结果
    cate_flag = 0
    start = -1
    end = -1
    ner_result = []
    for idx, cate in enumerate(label_result):
        if cate == 0 and cate_flag != 0:
            ner_result.append((CATEGORY[cate_flag - 1], start, end + 1, sent[start:end + 1]))
            start, end = idx, idx
            cate_flag = cate
        if cate != 0:
            if cate_flag == 0:  # 某一个词的开始
                start, end = idx, idx
                cate_flag = cate
            elif cate_flag != 0 and cate == cate_flag:  # 在某一个词的中间
                end = idx
            elif cate_flag != 0 and cate != cate_flag:  # 已经是另外一个词
                ner_result.append((CATEGORY[cate_flag - 1], start, end + 1, sent[start:end + 1]))
                start, end = idx, idx
                cate_flag = cate
    return label_result, ner_result
