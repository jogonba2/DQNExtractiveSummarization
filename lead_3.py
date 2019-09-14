# RESULTADO: {'ROUGE-1-P': 0.3359, 'ROUGE-1': 0.4024, 'ROUGE-2-P': 0.14784, 'ROUGE-2': 0.17705, 'ROUGE-L-P': 0.30449, 'ROUGE-L': 0.3645} #

from Utils import Utils
from CNNDailyMailEnvironment import CNNDailyMailEnvironment
import numpy as np
import rouge
from pythonrouge.pythonrouge import Pythonrouge


# Environment #
train_path = "./CNNDMCorpus/test.csv"

# Padding #
MAX_LEN_DOC = 100 #20 # 20

# Training #
N_SAMPLES_PER_EPISODE = 512 #11490


if __name__ == "__main__":
    utils = Utils()
    env = CNNDailyMailEnvironment(train_path, -1)
    env_gen = env.get_environment_sample()
    scores_1, scores_2, scores_3 = [], [], []
    gen_summaries = []
    summaries = []
    avg_score = 0.
    for i in range(N_SAMPLES_PER_EPISODE):
        if i % 200 == 0: print(i)
        #print("Sample: %d" % i)
        document, summary = next(env_gen)
        document = document
        summary = summary
        spl_document = utils.sentence_split(document, max_len = MAX_LEN_DOC)
        gen_summary = spl_document[:3] # k=3 frases
        gen_summaries.append(gen_summary)
        spl_summary = utils.sentence_split(summary, max_len = 9999)
        summaries.append([spl_summary])

rouge = Pythonrouge(summary_file_exist=False, delete_xml = True,
                    summary=gen_summaries, reference=summaries,
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                    f_measure_only=True, stemming=True, stopwords=False,
                    word_level=True, length_limit=False)

sc = rouge.calc_score()
print(sc)
print((sc["ROUGE-1"] + sc["ROUGE-2"] + sc["ROUGE-L"]) / 3)
