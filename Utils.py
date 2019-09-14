#from bert_serving.client import BertClient
from gensim.models import Word2Vec
import numpy as np


class Utils:

    def __init__(self):
        self.w2v_path = "./w2v_models/cnndm_w2v.model"
        self.w2v = Word2Vec.load(self.w2v_path)
        #self.bc = BertClient()

    # Devuelve el documento partido por frases (separadas por "\n") hasta un limite de max_len frases #
    def sentence_split(self, x, max_len):
        return x.split("\n")[:max_len]

    def sentence_embedding(self, x):
        r = np.zeros((len(x), self.w2v.vector_size))
        for i in range(len(x)):
            sline = x[i].split()
            c = 0.
            for j in range(len(sline)):
                if sline[j] in self.w2v:
                    r[i] += self.w2v[sline[j]]
                    c += 1.
            if c != 0:
                r[i] /= c
        return r

    def avg_doc_embedding(self, x):
        emb = self.sentence_embedding(x)
        return r.mean()

    # Arreglar padding #
    def compose(self, spl_document, action_seq):
        composed_summ = []
        for i in range(len(action_seq)):
            if action_seq[i] == 1:
                composed_summ.append(spl_document[i])
        return composed_summ
