from model import BiLstmCrfTrainer
from evaluate import merge_preds4ner, f1_score4ner
from utils import logger_init
from preprocess import CATEGORY, DataProcessor

from keras.callbacks import EarlyStopping
import numpy as np
import argparse
import datetime
import pickle

logger = logger_init()


def train(args):
    logger.info("***** Start *****")
    logger.info("Step1 : Loading and preparing Data")

    emb_matrix = pickle.load(open(args.emb_matrix_output_path, 'rb'))
    trainset = pickle.load(open('./saved_model_files/trainset.pkl', 'rb'))
    valset = pickle.load(open('./saved_model_files/valset.pkl', 'rb'))

    train_processor = DataProcessor(trainset).data4NER(window=args.window_size, pad=args.expend_size)
    val_processor = DataProcessor(valset).data4NER(window=args.window_size, pad=args.expend_size)

    train_X, train_Y = train_processor.get_ner_data()
    train_Y = np.expand_dims(train_Y, -1)

    val_X, val_Y = val_processor.get_ner_data()
    val_Y = np.expand_dims(val_Y, -1)

    logger.info("Step1 : Loading Done!")

    if args.model_type == "bilstm-crf":
        model = BiLstmCrfTrainer(category_count=len(CATEGORY) + 1,
                                 seq_len=train_X.shape[1],
                                 lstm_units=args.lstm_units,
                                 vocab_size=emb_matrix.shape[0],
                                 emb_matrix=emb_matrix).build()
        early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=args.stop_patience, mode='max')
        logger.info("Step2 : Start Training")
        logger.info("***** Pramaters *****")
        logger.info("  lstm_units = %s", args.lstm_units)
        logger.info("  BatchSize = %d", args.batch_size)
        model.fit(train_X, train_Y, batch_size=args.batch_size,
                  epochs=args.epoch,
                  class_weight="auto",
                  callbacks=[early_stopping],
                  validation_data=(val_X, val_Y,),
                  verbose=2
                  )

        time = datetime.datetime.now()
        model.save(filepath="./saved_model_files/bi_lstm_crf_{}_{}_{}_{}.h5".format(str(time.month),
                                                                                    str(time.day),
                                                                                    str(time.hour),
                                                                                    str(time.minute)), overwrite=True)
        logger.info("Step3 : Start Testing")
        testset = pickle.load(open('./saved_model_files/testset.pkl', 'rb'))
        testprocessor = DataProcessor(testset).data4NER(window=args.window_size, pad=args.expend_size)
        test_X, _ = testprocessor.get_ner_data()

        preds = model.predict(test_X, batch_size=args.batch_size, verbose=2)

        pre_docs = merge_preds4ner(testset, testprocessor, preds)
        source_docs = testset.docs

        f1, prediction, recall, _ = f1_score4ner(pre_docs, source_docs, 'all')
        logger.info("【严格相交】F1:%.4f  P:%.4f  R:%.4f", f1, prediction, recall)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="bilstm-crf", required=True, help="bilstm-crf or bilstm-lan")
    parser.add_argument("--window_size", type=int, default=70, required=True, help="Sliding Window size of Input")
    parser.add_argument("--expend_size", type=int, default=10, required=True, help="Expending size for Sliding Window")
    parser.add_argument("--lstm_units", type=str, default='128,128', required=True, help="Units counts for each Step")
    parser.add_argument("--emb_matrix_output_path", type=str, default="./saved_model_files/emb_matrix.pkl",
                        required=True, help="output path of word2idx")
    parser.add_argument("--stop_patience", type=int, default=2, required=True, help="How many epoch for early_stopping")

    parser.add_argument("--batch_size", type=int, default=64, required=True, help="BatchSize for Train")
    parser.add_argument("--epoch", type=int, default=50, required=True, help="Epoch for Train")

    args = parser.parse_args()

    args.lstm_units = [int(units) for units in str(args.lstm_units).split(',')]

    train(args)


if __name__ == "__main__":
    main()
