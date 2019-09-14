import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import random
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from QAgent import QAgent
from CNNDailyMailEnvironment import CNNDailyMailEnvironment
from Utils import Utils
import time
import multiprocessing as mp

def heuristic_exploration(spl_doc, required_padding, spl_summary, max_len, env, k):
    r = np.zeros(max_len, dtype = "int32")
    doc_len = len(spl_doc)
    rouge_sentences = np.zeros(doc_len)
    for i in range(doc_len):
        rouge_sentences[i] = env.get_reward([spl_doc[i]], spl_summary)
    idxs = np.argsort(rouge_sentences)[::-1][:k]
    r[idxs + required_padding] = 1
    return r

def worker(x):
    i, sent, spl_summary = x
    rouge_sents[i] = env.get_reward([sent], spl_summary)

def init_globals(rouge_sentences):
    global rouge_sents
    rouge_sents = rouge_sentences

def heuristic_exploration_parallel(spl_doc, required_padding, spl_summary, max_len, env, k):
    r = np.zeros(max_len, dtype = "int32")
    doc_len = len(spl_doc)
    rouge_sentences = mp.Array("d", np.zeros(doc_len), lock=False)
    num_workers = mp.cpu_count()
    with mp.Pool(num_workers, initializer = init_globals, initargs=(rouge_sentences,)) as pool:
        pool.map(worker, [(i, spl_doc[i], spl_summary) for i in range(doc_len)])
    pool.close()
    pool.join()
    idxs = np.argsort(rouge_sentences)[::-1][:k]
    r[idxs + required_padding] = 1
    return r

train_path = "./CNNDMCorpus/train.csv"

# Padding #
MAX_LEN_DOC = 30 #30

# Training #
EPISODES = 10000000
MEMORY_SIZE = 512
N_SAMPLES_PER_EPISODE = 128 #128 #128
BATCH_SIZE = 8 #8 #2
COPY_TARGET_WEIGHTS = 64 # SAMPLES
TYPE_REWARD = "rouge-avg"

# Dims #
INPUT_DIMS = 300 #300
ACTION_DIMS = 2
LSTM_HIDDEN_DIMS = 512 #512


if __name__ == "__main__":
    agent = QAgent(INPUT_DIMS, ACTION_DIMS, MEMORY_SIZE, BATCH_SIZE,
                     LSTM_HIDDEN_DIMS, MAX_LEN_DOC)
    utils = Utils()
    env = CNNDailyMailEnvironment(train_path, TYPE_REWARD)
    env_gen = env.get_environment_sample()
    best_episode_score = float("-inf")

    for e in range(EPISODES):

        spl_documents, spl_summaries = [], []
        repr_documents = []
        len_documents = []

        for i in range(N_SAMPLES_PER_EPISODE):
            document, summary = next(env_gen)
            spl_document = utils.sentence_split(document, max_len = MAX_LEN_DOC)
            spl_summary = utils.sentence_split(summary, max_len = 9999)
            repr_document = utils.sentence_embedding(spl_document)
            if len(spl_summary)==0 or len(spl_document)==0: continue
            spl_documents.append(spl_document)
            spl_summaries.append(spl_summary)
            repr_documents.append(repr_document)
            len_documents.append(len(spl_document))

        n_samples = len(spl_documents)
        repr_documents = pad_sequences(repr_documents, maxlen = MAX_LEN_DOC, dtype = "float32")
        init_c_state = np.zeros((n_samples, LSTM_HIDDEN_DIMS)) + 1e-16
        init_h_state = np.zeros((n_samples, LSTM_HIDDEN_DIMS)) + 1e-16
        lstm_h_states, lstm_c_states = agent.reader.predict([repr_documents,
                                                             init_h_state,
                                                             init_c_state])
        rewards = []
        for i in range(n_samples):
            try:
                required_padding = max(MAX_LEN_DOC - len_documents[i], 0)
                inp_1 = repr_documents[i].reshape((1, MAX_LEN_DOC, INPUT_DIMS))
                inp_2 = lstm_h_states[i].reshape((1, LSTM_HIDDEN_DIMS))
                inp_3 = lstm_c_states[i].reshape((1, LSTM_HIDDEN_DIMS))


                if np.random.rand() < agent.exploration_rate:
                    actions = heuristic_exploration_parallel(spl_documents[i], required_padding, spl_summaries[i], MAX_LEN_DOC, env, k = 3)
                else:
                    actions = agent.get_action(inp_1, inp_2, inp_3, required_padding)

                unpad_actions = actions[required_padding:]
                gen_summary = utils.compose(spl_documents[i], unpad_actions)
                rewards.append(env.get_reward(gen_summary, spl_summaries[i]))
                agent.remember(repr_documents[i], lstm_h_states[i], lstm_c_states[i], actions, rewards[-1], required_padding)
                agent.train_model()

            except Exception as exception:
                print(exception)
                continue

        episode_score = np.array(rewards).mean()
        if episode_score > best_episode_score:
            # Guardar modelos a partir de que el ratio de exploración este a menos de 0.1
            if agent.exploration_rate < 0.1:
                print("Saved model")
                agent.reader.save_weights("best_reader_weights.h5")
                agent.model.save_weights("best_model_weights.h5")
                best_episode_score = episode_score

        agent.reader.save_weights("reader_weights.h5")
        agent.model.save_weights("model_weights.h5")

        print("Avg on episode: %d , %.6f" % (e, episode_score))
