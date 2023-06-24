import pandas as pd
import argparse
import stanza
import evaluate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import seaborn as sns
from tqdm import tqdm
import ast

from typing import Callable

sns.set_context("talk")
sns.set_style("whitegrid")

def make_lemmatize(nlp: stanza.Pipeline) -> Callable:
    def lemmatize(sample: str) -> str:
        lemma = nlp(sample["translation"][args.lang])
        try:
            lemma_str =  " ".join([word.lemma for sent in lemma.sentences for word in sent.words])
        except:
            print("couldn't lemmatize", sample['translation'][args.lang])
            lemma_str = sample["translation"][args.lang]
        sample["translation_lemma"] = lemma_str
        return sample
    return lemmatize

def process_percentiles_data(lang: str, lang2: str,  model: str, idiomatic_translations: pd.DataFrame) -> None:
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    tqdm.pandas()

    # get base forms
    nlp = stanza.Pipeline(lang=lang2, processors="tokenize,pos,lemma")
    lemma_fn = make_lemmatize(nlp)
    print("Lemmatizing translations")
    idiomatic_translations["lemmatized"] = idiomatic_translations["source"].progress_apply(lambda x: lemma_fn({"translation": {args.lang: x}})["translation_lemma"])
    print("Lemmatizing idioms")
    idioms_lst = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{args.lang}'")["idiom"].tolist()]
    lemmatized_idioms = [nlp(x) for x in tqdm(idioms_lst)]
    idioms = set()
    for i, l in enumerate(lemmatized_idioms):
        sep = "" if lang == "jp" else " "
        lemma_idiom = sep.join([word.lemma for sent in l.sentences for word in sent.words])
        idioms.add(lemma_idiom)

    idioms_in_sents = []
    print("going through sentences")
    for i in tqdm(range(len(idiomatic_translations))):
        idiom_in_sent = []
        for idiom in idioms:
            if idiom in idiomatic_translations.iloc[i]["lemmatized"]:
                idiom_in_sent.append(idiom)
        idioms_in_sents.append(idiom_in_sent)
    idiomatic_translations["idioms_in_sents"] = idioms_in_sents
    freq_in_opensubtitles = pd.read_csv(f"./data/idiom_vs_literalized_selection/large_idiom_set_{args.lang}_value_counts.csv")
    # exclude double idiom sentences for now
    # exclude anything with a comma in Unnamed:0 column
    freq_in_opensubtitles = freq_in_opensubtitles[~freq_in_opensubtitles["Unnamed: 0"].str.contains(",")]
    total_idioms = freq_in_opensubtitles["contains_idioms"].sum()

    percentiles = [0.2, 0.4, 0.6, 0.8]
    freq_percentiles = [int(freq_in_opensubtitles["contains_idioms"].quantile(q=x)) for x in percentiles]
    bleu_by_percentile = []
    bert_by_percentile = []
    rouge_by_percentile = []
    meteor_by_percentile = []

    for p in freq_percentiles:
        print(f"idioms in the {percentiles[freq_percentiles.index(p)]} percentile (appear in {p}+ sentences))")

        if p == freq_percentiles[-1]:
            in_percentile = freq_in_opensubtitles.loc[(freq_in_opensubtitles["contains_idioms"] >= p)]
        else:
            in_percentile = freq_in_opensubtitles.loc[(freq_in_opensubtitles["contains_idioms"] >= p) & (freq_in_opensubtitles["contains_idioms"] < freq_percentiles[freq_percentiles.index(p) + 1])]
        in_percentile_idioms = in_percentile["Unnamed: 0"].tolist()
        translations_current_percentiles = idiomatic_translations.loc[idiomatic_translations["idioms_in_sents"].apply(lambda x: any(item in in_percentile_idioms for item in x))]
        freq = []
        for i in range(len(translations_current_percentiles)):
            curr_idioms = translations_current_percentiles.iloc[i]["idioms_in_sents"]
            for curr_idiom in curr_idioms:
                try:
                    freq.append(freq_in_opensubtitles.loc[freq_in_opensubtitles["Unnamed: 0"] == curr_idiom]["contains_idioms"].values[0])
                    break
                except:
                    continue
        translations_current_percentiles["freq"] = freq

        hyps = [x.strip() for x in translations_current_percentiles["hypothesis"].tolist()]
        refs = [[x[0].strip()] for x in translations_current_percentiles["references"].tolist()]

        bleu_score_base = bleu.compute(predictions=hyps, references=refs)
        bert_score_base = bertscore.compute(predictions=hyps, references=refs, lang="en")
        bert_score_base = sum(bert_score_base["f1"]) / len(bert_score_base["f1"])
        rouge_score_base = rouge.compute(predictions=hyps, references=refs)
        results_meteor_base = meteor.compute(predictions=hyps, references=refs)

        bleu_by_percentile.append(bleu_score_base["bleu"])
        bert_by_percentile.append(bert_score_base)
        rouge_by_percentile.append(rouge_score_base["rouge1"])
        meteor_by_percentile.append(results_meteor_base["meteor"])

        print(f"BLEU score base: {bleu_score_base}")
        print(f"BERT score base: {bert_score_base}")
        print(f"ROUGE score base: {rouge_score_base}")
        print(f"METEOR score base: {results_meteor_base}")

        translations_current_percentiles.sort_values(by="freq", ascending=False, inplace=True)
        translations_current_percentiles.to_csv(f"data/freq/{lang}_{model}_percentile_{percentiles[freq_percentiles.index(p)]}.csv", index=False)

    return bleu_by_percentile, bert_by_percentile, rouge_by_percentile, meteor_by_percentile

def plot_metrics_vs_freq(percentiles, bleu_by_percentile, bert_by_percentile, rouge_by_percentile, meteor_by_percentile, lang, model, cond2=None, output=None):
    assert len(percentiles) == len(bleu_by_percentile)
    assert len(percentiles) == len(bert_by_percentile)
    assert len(percentiles) == len(rouge_by_percentile)
    assert len(percentiles) == len(meteor_by_percentile)

    data = pd.DataFrame({
    'Percentiles': percentiles * 4, 
    'Metric': ['BLEU'] * len(percentiles) + ['BERTScore'] * len(percentiles) +
              ['ROUGE'] * len(percentiles) + ['METEOR'] * len(percentiles),
    'Score': bleu_by_percentile + bert_by_percentile + rouge_by_percentile + meteor_by_percentile
    })

    if cond2 is not None:
        data["Condition"] = cond2 * 4
        unique_vals = ["base", "upweight+knn"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    metrics = ['BLEU', 'BERTScore', 'ROUGE', 'METEOR']
    colors = ['blue', 'green', 'red', 'orange']
    additional_colors = ["royalblue", "limegreen", "firebrick", "darkorange"]

    for ax, metric, color in zip(axs.flat, metrics, colors):
        if cond2 is not None:
            cond1_data = data[(data['Metric'] == metric) & (data["Condition"] == unique_vals[0])]
            cond2_data = data[(data['Metric'] == metric) & (data["Condition"] == unique_vals[1])]
            sns.lineplot(data=cond1_data, x='Percentiles', y='Score', ax=ax, color=color, linestyle="-", label=unique_vals[0])
            sns.lineplot(data=cond2_data, x='Percentiles', y='Score', ax=ax, color=additional_colors[metrics.index(metric)], linestyle='--', label=unique_vals[1])
        else:
            sns.lineplot(data=data[data['Metric'] == metric], x='Percentiles', y='Score', ax=ax, color=color)

        ax.set_title(metric)

    # Label x-axis
    for ax in axs[-1, :]:
        ax.set_xlabel('Percentile of idioms in OS')

    # Label y-axis
    for ax in axs[:, 0]:
        ax.set_ylabel('Metric Score')

    custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-'),  # Line style for first line
                Line2D([0], [0], color='black', lw=2, linestyle='--')]  # Line style for second line

    fig.legend(custom_lines, ["base", "upweight+knn"], loc='upper center', bbox_to_anchor=(0.5, 1.1), bbox_transform=plt.gcf().transFigure)

    plt.tight_layout()
    if args.output is None:
        plt.savefig(f"data/freq/{lang}_{model}_percentile.png")
        plt.savefig(f"data/freq/{lang}_{model}_percentile.pdf")
    else:
        plt.savefig(f"{args.output}.png")
        plt.savefig(f"{args.output}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare base form to conjugated form idioms")
    parser.add_argument("-l", "--lang", help="language", choices=["fr", "fi", "jp"], required=True)
    parser.add_argument("-m", "--model", help="model", choices=["google", "deepl", "deltalm"], required=True)
    parser.add_argument("-f", "--format", default="csv", help="data format", choices=["csv", "sys"])
    parser.add_argument("-i", "--input_dir", help="input directory (needed for sys format)", nargs="+")
    parser.add_argument("-o", "--output", help="output directory")
    args = parser.parse_args()
    
    assert len(args.input_dir) == 2, "need two input directories for comparative sys format"

    lang2 = args.lang if args.lang != "jp" else "ja"
    if args.format == "csv":
        translations_df = pd.read_csv(f"translation_comparison_{args.lang}_{args.model}.csv")
        idiomatic_translations = translations_df[translations_df["type"] == "idiomatic"]

        bleu_by_percentile, bert_by_percentile, rouge_by_percentile, meteor_by_percentile = [], [], [], []
        percentiles = [0.2, 0.4, 0.6, 0.8]
        plot_metrics_vs_freq(percentiles, bleu_by_percentile, bert_by_percentile, rouge_by_percentile, meteor_by_percentile, args.lang, args.model)
    elif args.format == "sys":
        all_src = []
        all_hyp = []
        all_ref = []
        conds = []
        for i, input_dir in enumerate(args.input_dir):
            src_filename = os.path.join(input_dir, "srcs.txt")
            hyp_filename = os.path.join(input_dir, "hyps.txt")
            ref_filename = os.path.join(input_dir, "tgts.txt")

            with open(src_filename, "r") as f:
                srcs = f.readlines()
            with open(hyp_filename, "r") as f:
                hyps = f.readlines()
            with open(ref_filename, "r") as f:
                refs = f.readlines()
            refs = [[x] for x in refs]

            all_src.extend(srcs)
            all_hyp.extend(hyps)
            all_ref.extend(refs)
            conds.extend([i] * len(srcs))

        idiomatic_translations = pd.DataFrame({"source": all_src, "hypothesis": all_hyp, "references": all_ref, "cond": conds})

        bleu_by_percentile0, bert_by_percentile0, rouge_by_percentile0, meteor_by_percentile0 = process_percentiles_data(args.lang, lang2, args.model, idiomatic_translations.loc[idiomatic_translations["cond"] == 0])
        bleu_by_percentile1, bert_by_percentile1, rouge_by_percentile1, meteor_by_percentile1 = process_percentiles_data(args.lang, lang2, args.model, idiomatic_translations.loc[idiomatic_translations["cond"] == 1])

        bleu_by_percentile = bleu_by_percentile0 + bleu_by_percentile1
        bert_by_percentile = bert_by_percentile0 + bert_by_percentile1
        rouge_by_percentile = rouge_by_percentile0 + rouge_by_percentile1
        meteor_by_percentile = meteor_by_percentile0 + meteor_by_percentile1

        percentiles = [0.2, 0.4, 0.6, 0.8] * 2
        plot_metrics_vs_freq(percentiles, bleu_by_percentile, bert_by_percentile, rouge_by_percentile, meteor_by_percentile, args.lang, args.model, cond2=["base"] * 4 + ["upweight+knn"] * 4, output=args.output)