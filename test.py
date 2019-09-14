import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import random
from keras.layers import LSTM, Dense, Activation, Input, Lambda, Concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from collections import deque
from keras.preprocessing.sequence import pad_sequences
from QAgent import QAgent
from CNNDailyMailEnvironment import CNNDailyMailEnvironment
from Utils import Utils
import time
from pythonrouge.pythonrouge import Pythonrouge


# Environment #
train_path = "./CNNDMCorpus/test.csv"
reader_path = "./reader_weights.h5"
model_path = "./model_weights.h5"

# Padding #
MAX_LEN_DOC = 30 #20 # 20

# Training #
MEMORY_SIZE = None
BATCH_SIZE = None

# Dims #
INPUT_DIMS = 300 #300
ACTION_DIMS = 2
LSTM_HIDDEN_DIMS = 512 #512


if __name__ == "__main__":
    agent = QAgent(INPUT_DIMS, ACTION_DIMS, MEMORY_SIZE, BATCH_SIZE,
                     LSTM_HIDDEN_DIMS, MAX_LEN_DOC)
    agent.load_weights(reader_path, model_path)
    utils = Utils()
    env = CNNDailyMailEnvironment(train_path, None)
    env_gen = env.get_environment_sample() # Usar el sample_test() para la evaluacion final, y sample() para pruebas rápidas (las primeras muestras del test no dan buenos resultados y engañan) #
    best_episode_score = float("-inf")

    gen_summaries = []
    references = []
    c = 0
    for document, summary in env_gen:
        spl_document = utils.sentence_split(document, max_len = MAX_LEN_DOC)
        spl_summary = utils.sentence_split(summary, max_len = 9999)
        if len(spl_summary)==0: continue
        repr_document = utils.sentence_embedding(spl_document)
        repr_document = pad_sequences([repr_document], maxlen = MAX_LEN_DOC, dtype = "float32")
        init_c_state = np.zeros((LSTM_HIDDEN_DIMS,)) + 1e-16
        init_h_state = np.zeros((LSTM_HIDDEN_DIMS,)) + 1e-16
        lstm_h_state, lstm_c_state = agent.reader.predict([repr_document,
                                                           np.array([init_h_state]),
                                                           np.array([init_c_state])])
        len_document = len(spl_document)
        required_padding = max(MAX_LEN_DOC - len_document, 0)
        action_seq = agent.get_action_test(repr_document, lstm_h_state, lstm_c_state, required_padding)
        action = action_seq[required_padding:]
        print(action)
        gen_summary = utils.compose(spl_document, action)
        gen_summaries.append(gen_summary)
        references.append([spl_summary])
        #print(spl_document)
        #print(action_seq)
        #print(gen_summary)
        #print(spl_summary)
        #print("---------------------\n\n")
        if c != 0 and c % 300 == 0: break #print(c)
        c += 1

rouge = Pythonrouge(summary_file_exist=False, delete_xml = True,
                    summary=gen_summaries, reference=references,
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                    f_measure_only=True, stemming=True, stopwords=False,
                    word_level=True, length_limit=False)

print(rouge.calc_score())
