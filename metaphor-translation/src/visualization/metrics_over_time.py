import argparse
import evaluate
import pandas as pd
import os
import stanza
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.models.evaluate_translations import get_paraphrase_rate
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make a plot of metrics over time in training")
    parser.add_argument("-d", "--checkpoint_dir", help="directory containing checkpoints labelled with number of steps", required=True)
    parser.add_argument("-l", "--lang", help="language", choices=["fr", "fi", "jp"], required=True)
    args = parser.parse_args()

    bleu = evaluate.load("bleu")
    idiom_df = pd.read_csv("./data/external/all_idioms_naive_translation.csv").query(f"lang == '{args.lang}'")
    lang2 = "ja" if args.lang == "jp" else args.lang
    nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)

    timesteps = []
    bleu_scores = []
    paraphrase_scores = []

    for filename in os.listdir(args.checkpoint_dir):
        filepath = os.path.join(args.checkpoint_dir, filename)
        if os.path.isdir(filepath):
            print(filepath)
            epoch_num = int(filename.split("_")[1])
            hyps = [line.rstrip() for line in open(f"{filepath}/hyps.txt", "r")]
            refs = [line.rstrip() for line in open(f"{filepath}/tgts.txt", "r")]
            srcs = [line.rstrip() for line in open(f"{filepath}/srcs.txt", "r")]
            df = pd.DataFrame({"sent": srcs, "reference": refs, "translation": hyps})

            paraphrase_rate, other_scores = get_paraphrase_rate(df, idiom_df, nlp, args.lang, hyps, refs, bleu=bleu)
            bleu_score = other_scores[0]["bleu"]

            timesteps.append(epoch_num)
            paraphrase_scores.append(paraphrase_rate)
            bleu_scores.append(bleu_score)
    timesteps, bleu_scores, paraphrase_scores = zip(*sorted(zip(timesteps, bleu_scores, paraphrase_scores)))
    info_df = pd.DataFrame({"steps": timesteps, "BLEU": bleu_scores, "Paraphrase rate": paraphrase_scores})
    plt.plot(timesteps, bleu_scores, label="BLEU")
    plt.plot(timesteps, paraphrase_scores, label="Paraphrase rate")
    plt.legend()
    plt.savefig(f"{args.lang}_over_time_try.png")