from gensim.models import Word2Vec
from CNNDailyMailEnvironment import CNNDailyMailEnvironment

train_path = "./CNNDMCorpus/train.csv" # CNNDailyMail
w2v_name = "./w2v_models/cnndm_w2v.model"
n_samples = 287228

env = CNNDailyMailEnvironment(train_path, -1)
env_gen = env.get_environment_sample()
all_text = []

for i in range(n_samples):
    if i % 1000 == 0: print(i)
    article, summary = next(env_gen)
    if article == None or summary == None:
        continue
    sarticle = article.split()
    ssummary = summary.split()
    all_text.append(sarticle)
    all_text.append(ssummary)

print(all_text[0])

w2v = Word2Vec(all_text, size=300, min_count=5, window=5, workers=12, sg=1, hs=0, negative=5)

w2v.save(w2v_name)
