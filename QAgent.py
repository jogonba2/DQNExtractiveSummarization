# PER: https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814
# PER: https://medium.com/arxiv-bytes/summary-prioritized-experience-replay-e5f9257cef2d

import numpy as np
import random
from keras.layers import LSTM, Dense, Input, Lambda, Concatenate, Masking, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from collections import deque
from keras.preprocessing.sequence import pad_sequences
import time
import tensorflow as tf


class QAgent:

    def __init__(self, input_dims, action_dims, memory_size, batch_size,
                 lstm_dims, max_len_doc):

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.memory_size = memory_size
        self.gamma = 0.99
        self.lr = 0.00001 # Parece importante
        self.batch_size = batch_size
        self.exploration_max = 1.0
        self.exploration_min = 0.05
        self.exploration_decay = 0.9995 # Parece importante la exploracion, intentar subir el minimo y que explore más de continuo
        self.exploration_rate = self.exploration_max

        self.max_len_doc = max_len_doc

        self.memory = deque(maxlen = self.memory_size)

        self.lstm_dims = lstm_dims
        self.reader, self.model = self.build_models()

        #_, self.target = self.build_models()
        #self.target.set_weights(self.model.get_weights())

    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = tf.keras.backend.abs(error) < clip_delta
        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
        return tf.where(cond, squared_loss, linear_loss)

    def build_models(self):

        doc_state = Input(shape=(self.max_len_doc, self.input_dims))
        #masked_doc_state = Masking(mask_value = 0.)(doc_state)
        state_h = Input(shape=(self.lstm_dims,))
        state_c = Input(shape=(self.lstm_dims,))

        lstm = LSTM(self.lstm_dims, activation="tanh", name="lstm_1",
                    return_sequences=True, return_state=True)

        o1, lstm_state_h, lstm_state_c = lstm(doc_state,
                                              initial_state=[state_h, state_c])

        output = Dense(self.action_dims, activation="linear")(o1)

        reader = Model(inputs=[doc_state, state_h, state_c],
                       outputs=[lstm_state_h, lstm_state_c])

        model = Model(inputs=[doc_state, state_h, state_c],
                      outputs=output)

        model.compile(loss = self.huber_loss, optimizer = Adam(lr = self.lr))

        return reader, model

    def get_action(self, doc_state, state_h, state_c, required_padding):
        #if np.random.rand() < self.exploration_rate:
        #    return np.random.randint(low = 0, high = 2, size = self.max_len_doc) # - required_padding)

        p = self.model.predict([doc_state, state_h, state_c])[0]
        return p.argmax(axis = -1) #[required_padding:]

    def get_action_test(self, doc_state, state_h, state_c, required_padding):
        p = self.model.predict([doc_state, state_h, state_c])[0]
        return p.argmax(axis = -1) #[required_padding:]

    def remember(self, doc_state, state_h, state_c, action_seq, reward, required_padding):
        self.memory.append((doc_state, state_h, state_c, action_seq, reward, required_padding))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        doc_states, states_h, states_c, action_seqs, rewards, required_padding = map(np.array, zip(*batch))
        targets = rewards
        q_vals = self.model.predict([doc_states, states_h, states_c], batch_size = self.batch_size)

        for i in range(len(q_vals)):
            for j in range(self.max_len_doc):
                q_vals[i][j][action_seqs[i][j]] = targets[i]


        self.model.fit([doc_states, states_h, states_c],
                        q_vals,
                        batch_size = self.batch_size,
                        epochs = 1,
                        verbose = 0)

        # Aunque se entrene en cuanto se pueda crear un batch, reducirá la exploración cuando llene la memoria #
        if not (len(self.memory) < self.memory_size):
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def set_target_weights(self):
        self.target.set_weights(self.model.get_weights())

    def load_weights(self, path_reader, path_model):
        self.reader.load_weights(path_reader)
        self.model.load_weights(path_model)
