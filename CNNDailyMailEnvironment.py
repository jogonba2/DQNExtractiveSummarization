# ROUGE: https://github.com/tagucci/pythonrouge
import pandas as pd
from pythonrouge.pythonrouge import Pythonrouge
from ast import literal_eval
from random import randint


class CNNDailyMailEnvironment:

    def __init__(self, csv_file, type_reward):
        self.csv_file = csv_file
        self.type_reward = type_reward # \in {rouge-1, rouge-2, rouge-l, rouge-avg} #
        self.evaluator = Pythonrouge(summary_file_exist=False, delete_xml = True,
                    summary=[], reference=[],
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                    f_measure_only=True, stemming=True, stopwords=False,
                    word_level=True, length_limit=False)


    def get_environment_sample(self):
        aux_samples = []
        for chunk in  pd.read_csv(self.csv_file, sep='\s*\t\s*', lineterminator="\n", chunksize=20000, engine="python"):
            aux_samples.append(chunk)
        csv_samples = pd.concat(aux_samples, axis=0)
        del aux_samples
        n_samples = len(csv_samples)

        while True:
            for i in range(n_samples):
                idx = randint(0, n_samples - 1)
                yield (literal_eval(csv_samples.iloc[idx]["TEXT"]), literal_eval(csv_samples.iloc[idx]["SUMMARY"]))

    def get_statistics(self):
        avg_sents = 0.
        c = 0.
        for doc, _ in self.get_environment_sample_test():
            avg_sents += len(doc.split("\n"))
            c += 1.
            if c % 1000 == 0: print(c)
        print(avg_sents / c)

    def get_environment_sample_test(self):
        aux_samples = []
        for chunk in  pd.read_csv(self.csv_file, sep='\s*\t\s*', lineterminator="\n", chunksize=20000, engine="python"):
            aux_samples.append(chunk)

        csv_samples = pd.concat(aux_samples, axis=0)
        del aux_samples

        for i in range(len(csv_samples)):
            yield literal_eval(csv_samples.iloc[i]["TEXT"]), literal_eval(csv_samples.iloc[i]["SUMMARY"])

    def get_reward(self, gen_summary, reference):
       #try:
       if len(gen_summary) == 0:
           return -1.
       if len(reference) == 0:
           return 1e-16

       self.evaluator.summary = [gen_summary]
       self.evaluator.reference = [[reference]]
       rouge_score = self.evaluator.calc_score()

       if self.type_reward == "rouge-1":
           reward = rouge_score["ROUGE-1'"]
       elif self.type_reward == "rouge-2":
           reward = rouge_score["ROUGE-2"]
       elif self.type_reward == "rouge-l":
           reward = rouge_score["rouge-L"]
       else:
           reward_1 = rouge_score["ROUGE-1"]
           reward_2 = rouge_score["ROUGE-2"]
           reward_3 = rouge_score["ROUGE-L"]
           #print("ROUGE-1: %.4f\nROUGE-2: %.4f\nROUGE-L: %.4f\n" % (reward_1, reward_2, reward_3))
           reward = (reward_1 + reward_2 + reward_3) / 3.

       return reward

       #except Exception as e:
       #    print("Error", e)
       #    print("Gen summary:", gen_summary)
       #    print("Summary:", reference)
       #    return 1e-16


    def get_full_reward(self, gen_summary, reference):
       try:
           if len(gen_summary) == 0:
               return 1e-16
           if len(reference) == 0:
               return 1e-16

           self.evaluator.summary = [gen_summary]
           self.evaluator.reference = [[reference]]
           rouge_score = self.evaluator.calc_score()

           if self.type_reward == "rouge-1":
               reward = rouge_score["ROUGE-1'"]
           elif self.type_reward == "rouge-2":
               reward = rouge_score["ROUGE-2"]
           elif self.type_reward == "rouge-l":
               reward = rouge_score["rouge-L"]
           else:
               reward_1 = rouge_score["ROUGE-1"]
               reward_2 = rouge_score["ROUGE-2"]
               reward_3 = rouge_score["ROUGE-L"]
               #print("ROUGE-1: %.4f\nROUGE-2: %.4f\nROUGE-L: %.4f\n" % (reward_1, reward_2, reward_3))
               reward = (reward_1 + reward_2 + reward_3) #/ 3.

           return reward

       except Exception as e:
           print("Error", e)
           print("Gen summary:", gen_summary)
           print("Summary:", reference)
           return 1e-16


if __name__ == "__main__":

    env = CNNDailyMailEnvironment("./CNNDMCorpus/train.csv", "-1")
    env.get_statistics()
