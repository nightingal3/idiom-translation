from datasets import load_dataset
from eaas import Config, Client
from tqdm import tqdm
import pickle
import pdb
import sys

client = Client(Config())

dataset = load_dataset("open_subtitles", lang1="en", lang2="fr")["train"]
#dataset = pickle.load(open("./translations_fr.p", "rb"))
#with open(sys.argv[1], 'r') as srcfile:
  #srcs = [x.strip() for x in srcfile]

#with open(sys.argv[2], 'r') as hypfile:
  #hyps = [x.strip() for x in hypfile]

inputs = []
srcs = []
hyps = []
for sample in tqdm(dataset):
  src = sample["translation"]["fr"]
  hyp = sample["translation"]["en"]
  #src = sample["source"]
  #hyp = sample["references"][0]
  srcs.append(src)
  hyps.append(hyp)

  inputs.append({'source': src, 'references': [hyp], 'hypothesis': hyp})

all_scores_and_sents = []
for i in range(len(inputs)//1000 + 1):
  print(i * 1000)
  metrics = ["comet_qe"]
  curr_inputs = inputs[i * 1000:min((i + 1) * 1000, len(inputs))]
  score_dic = client.score(curr_inputs, metrics=metrics)
  annotated_inputs = list(zip(score_dic['scores'][0]['sample'], srcs, hyps))
  annotated_inputs.sort()
  for score, src, hyp in annotated_inputs:
    all_scores_and_sents.append((score, src, hyp))

all_scores_and_sents.sort(key=lambda x: x[0])
with open("fr_cometqe.p", "wb") as f:
  pickle.dump(all_scores_and_sents, f)
  
for score, src, hyp in all_scores_and_sents:
  print(f'{score}\t{src}\t{hyp}')
