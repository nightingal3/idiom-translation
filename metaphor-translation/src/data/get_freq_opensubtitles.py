import argparse
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=["fr", "fi", "jp"])
    args = parser.parse_args()

    dataset = load_from_disk(f"open_subtitles_{args.lang}-filtered_lemma_dataset.hf")
    if "train" in dataset:
        dataset = dataset["train"]

    idiom_dataset = pd.read_csv(f"./data/test_sets_final/{args.lang}/idiomatic_all.csv")
    literal_dataset = pd.read_csv(f"./data/test_sets_final/{args.lang}/literal.csv")
    random_dataset = pd.read_csv(f"./data/test_sets_final/{args.lang}/random_sents_ted.csv")

    idioms = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{args.lang}'")["idiom"].tolist()]
    
    lang2 = args.lang if args.lang != "jp" else "ja"

    freq_in_opensubtitles = {idiom: 0 for idiom in idioms}

    for i in tqdm(range(len(dataset))):
        for idiom in idioms:
            if idiom in dataset[i]["translation_lemma"]:
                freq_in_opensubtitles[idiom] += 1

    freq_in_opensubtitles = {k: v for k, v in sorted(freq_in_opensubtitles.items(), key=lambda item: item[1], reverse=True)}
    freq_df = pd.DataFrame.from_dict(freq_in_opensubtitles, orient="index", columns=["freq"])
    freq_df.to_csv(f"./data/freq/opensubs_{args.lang}.csv")