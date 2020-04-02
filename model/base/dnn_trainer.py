from keras.optimizers import Adam


class NNTrainer:
    def __init__(self, category_count, seq_len, vocab_size, lstm_units, emb_matrix=None, optimizer=Adam()):
        self.category_count = category_count
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emb_matrix = emb_matrix
        self.lstm_units = lstm_units
        self.optimizer = optimizer

    def build(self):
        pass
