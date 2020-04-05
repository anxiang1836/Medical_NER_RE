"""
@Author      :   MaxMa
@Email       :   xingyangma@126.com
@File Name   :   bilstm_trainer.py
"""

from .base import NNTrainer, Attention
from keras.optimizers import Adam
from keras import Input
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Concatenate
from keras_contrib.layers import CRF
from keras.models import Model


# NER的BaseLine训练器
class BiLstmCrfTrainer(NNTrainer):

    def __init__(self, category_count, seq_len, vocab_size, lstm_units, emb_matrix=None, optimizer=Adam()):
        super().__init__(category_count, seq_len, vocab_size, lstm_units, emb_matrix, optimizer)

    def build(self):
        if self.emb_matrix is not None:
            embedding = Embedding(input_dim=self.vocab_size,
                                  output_dim=self.emb_matrix.shape[1],
                                  weights=[self.emb_matrix],
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.vocab_size, output_dim=128, trainable=True
                                  )  # mask_zero=True 这里给embedding的zero做mask

        model_input = Input(shape=(self.seq_len,), dtype="int32")
        embedding = embedding(model_input)
        dropout = Dropout(0.5)(embedding)
        lstm = LSTM(self.lstm_units, return_sequences=True)
        bi_lstm = Bidirectional(lstm)(dropout)
        bi_lstm = Dropout(0.5)(bi_lstm)
        crf = CRF(self.category_count, sparse_target=True)
        output = crf(bi_lstm)

        model = Model(model_input, output)
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        return model

        # NER的Attention层替换训练器


class BiLstm_Lan_Trainer(NNTrainer):
    def __init__(self, category_count, seq_len, vocab_size, lstm_units, emb_matrix=None, optimizer=Adam()):
        super().__init__(category_count, seq_len, vocab_size, lstm_units, emb_matrix, optimizer)

    def build(self):
        if self.emb_matrix is not None:
            embedding = Embedding(input_dim=self.vocab_size,
                                  output_dim=self.emb_matrix.shape[1],
                                  weights=[self.emb_matrix],
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.vocab_size, output_dim=128, trainable=True
                                  )  # mask_zero=True 这里给embedding的zero做mask

        model_input = Input(shape=(self.seq_len,), dtype="int32")
        embedding = embedding(model_input)
        x = Dropout(0.5)(embedding)

        for idx, param in enumerate(self.lstm_units):
            lstm = LSTM(param, return_sequences=True)
            if idx == len(self.lstm_units) - 1:
                # 表示是最后一层：
                x = Bidirectional(lstm)(x)
                x = Attention(self.category_count, is_last_layer=True)(x)
            else:
                x_1 = Bidirectional(lstm)(x)
                x_2 = Attention(self.category_count, is_last_layer=False)(x_1)
                x = Concatenate()([x_1, x_2])

        model = Model(model_input, x)

        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["acc"])
        return model
