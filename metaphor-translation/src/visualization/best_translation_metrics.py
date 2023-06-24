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
    parser.add_argument("-d", "--translation_dir", help="directory containing translations from the best checkpoint", default="/projects/tir5/users/mengyan3/unilm/deltalm/idiom_results_over_time/fi_no_test_best_idioms")
    parser.add_argument("-l", "--lang", help="language", choices=["fr", "fi", "jp"], required=True)
    args = parser.parse_args()

    idiom_df = pd.read_csv("./data/external/all_idioms_naive_translation.csv").query(f"lang == '{args.lang}'")
    lang2 = "ja" if args.lang == "jp" else args.lang
    nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)

    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    filepath = args.translation_dir
    hyps = [line.rstrip() for line in open(f"{filepath}/hyps.txt", "r")]
    refs = [line.rstrip() for line in open(f"{filepath}/tgts.txt", "r")]
    srcs = [line.rstrip() for line in open(f"{filepath}/srcs.txt", "r")]
    df = pd.DataFrame({"sent": srcs, "reference": refs, "translation": hyps})

    paraphrase_rate, other_scores = get_paraphrase_rate(df, idiom_df, nlp, args.lang, hyps, refs, bleu=bleu)
    bleu_score = other_scores[0]["bleu"]
    bert_score = bertscore.compute(predictions=hyps, references=refs, lang="en")
    bert_score = sum(bert_score["f1"]) / len(bert_score["f1"])
    rouge_score = rouge.compute(predictions=hyps, references=refs)
    results_meteor = meteor.compute(predictions=hyps, references=refs)
    print(f"Paraphrase rate: {paraphrase_rate}")
    print(f"BLEU: {bleu_score}")
    print(f"BERTScore: {bert_score}")
    print(f"ROUGE: {rouge_score}")
    print(f"METEOR: {results_meteor}")