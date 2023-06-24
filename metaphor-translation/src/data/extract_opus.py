import argparse
import logging
from datasets import load_dataset, load_from_disk
import pandas as pd
import json
from typing import Callable
import stanza
from tqdm import tqdm
import pdb
import os
from multiprocess import set_start_method
import torch

from src.utils.data_utils import find_idioms

set_start_method("spawn")

def make_lemmatize(nlp: stanza.Pipeline) -> Callable:
    def lemmatize(sample: str) -> str:
        lemma = nlp(sample["translation"][lang2])
        try:
            lemma_str =  " ".join([word.lemma for sent in lemma.sentences for word in sent.words])
        except:
            print("couldn't lemmatize", sample['translation'][lang2])
            lemma_str = sample["translation"][lang2]
        sample["translation_lemma"] = lemma_str
        return sample
    return lemmatize
    
def lemmatize_batch(sample: str) -> str:
    pdb.set_trace()
    lemma = [nlp(sample["translation"][i][lang2]) for i in range(len(sample["translation"]))]
    lemma_str = []
    for i, l in lemma:
        pdb.set_trace()
        try:
            lemma_str.append(" ".join([word.lemma for sent in l.sentences for word in sent.words]))
        except:
            print("couldn't lemmatize", l)
            lemma_str.append(sample["translation"][lang2])
    sample["translation_lemma"] = lemma_str
    
    return sample

def _lemmatize_batch(sample: str) -> str:
    for x in sample:
        lemma = nlp(x["translation"][lang2])
        try:
            lemma_str =  " ".join([word.lemma for sent in lemma.sentences for word in sent.words])
        except:
            print("couldn't lemmatize", sample['translation'][lang2])
            lemma_str = x["translation"][lang2]
        x["translation_lemma"] = lemma_str
    return sample
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract either idiomatic or random sentences from a dataset (must be on huggingface)")
    parser.add_argument("-d", "--dataset", help="dataset to use", choices=["opus_books", "open_subtitles"], default="opus_books")
    parser.add_argument("-l", "--lang", help="language to use (second language always EN)", choices=["fr", "fr-filtered", "jp", "jp-filtered", "fi", "fi-filtered"], default="fr")
    parser.add_argument("--lemma", help="use lemmatized idioms to match (also lemmatize dataset)", type=bool, default=True)
    parser.add_argument("-r", "--random", help="extract N random sentences, rather than idiomatic sentences.", type=int)
    parser.add_argument("-c", "--context_window", help="length of context window in sentences (for contextual MT)", type=int)
    parser.add_argument("-s", "--select_idioms", help="limit the idioms to a certain list", type=str, nargs="+")
    parser.add_argument("-i", "--index", help="index (for sharded dataset)", choices=list(range(0, 10)), type=int)
    args = parser.parse_args()

    SCRATCH_DIR = "/compute/tir-0-15/mengyan3/idiom-translation/data"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.dataset == "opus_books":
        lang2 = args.lang
        dataset = load_dataset("opus_books", f"en-{args.lang}")
    else:
        lang2 = args.lang if "jp" not in args.lang else "ja"
        if "filtered" in args.lang:
            lang2 = lang2[:2]
            dataset = load_dataset("json", data_files=f"data/processed/{lang2}_cometqe_comet_filtered.json")
            if args.index is not None:
                dataset = dataset["train"].shard(num_shards=10, index=args.index)
        else:
            dataset = load_dataset("open_subtitles", lang1="en", lang2=lang2)
            if args.index:
                dataset = dataset["train"].shard(num_shards=10, index=args.index)

    if args.random is not None:
        dataset_shuffled = dataset["train"].shuffle(seed=42)
        translation_pairs = {}

        for i, sample in enumerate(dataset_shuffled):
            if i > args.random:
                break
            translation_pairs[sample["id"]] = {
                "text": sample["translation"]["en"],
                "original_text": sample["translation"][lang2],
            }

        with open(f"data/external/{args.dataset}/{args.lang}_random.json", "w") as f:
            json.dump(translation_pairs, f, ensure_ascii=False, indent=4, separators=(",", ": "))
    else:
        if args.select_idioms:
            idioms = args.select_idioms
        else:
            base_lang = args.lang if "filtered" not in args.lang else args.lang.replace("-filtered", "")
            idioms = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{base_lang}'")["idiom"].tolist()]

        if args.lemma:
            nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)
            lemmatized_idioms = [nlp(x) for x in idioms]
            idioms = set()
            for i, l in enumerate(lemmatized_idioms):
                sep = "" if args.lang == "jp" else " "
                lemma_idiom = sep.join([word.lemma for sent in l.sentences for word in sent.words])
                idioms.add(lemma_idiom)

            lemma_fn = make_lemmatize(nlp)

        logging.info("Loading dataset...")
        if not os.path.exists(f"{args.dataset}_{args.lang}_lemma_dataset.hf"):
            logging.info("Dataset file doesn't exist yet, lemmatizing...")
            dataset_lemmatized = dataset.map(lemma_fn)
            dataset_lemmatized.save_to_disk(f"{args.dataset}_{args.lang}_{args.index}_lemma_dataset.hf")
        else:
            logging.info("Dataset file exists, loading from disk")
            dataset_lemmatized = load_from_disk(f"{args.dataset}_{args.lang}_lemma_dataset.hf")

        logging.info("Finding idiomatic sentences...")
        contains_idioms = []
        try:
            d_lemmatized = dataset_lemmatized["train"] 
        except:
            d_lemmatized = dataset_lemmatized

        for sample in tqdm(d_lemmatized):
            sample_text = sample["translation_lemma"] if args.lang != "jp" else "".join(sample["translation_lemma"].split())
            if any(idiom in sample_text for idiom in idioms):
                logging.debug(f"idiom found: {sample_text}")
                contains_idioms.append(sample)
        #contains_idioms = dataset_lemmatized["train"].filter(lambda x: any(idiom in x["translation_lemma"] for idiom in idioms))

        translation_pairs = {}

        logging.info("Collecting idiom data...")
        i = 0

        iter_dataset = contains_idioms["train"] if "train" in contains_idioms else contains_idioms
        for sample in tqdm(iter_dataset):
            if args.context_window:
                ind = int(sample["id"])
                prev_samples = [x[lang2] for x in dataset["train"].select(list(range(ind - args.context_window, ind)))["translation"]]
                next_samples = [x[lang2] for x in dataset["train"].select(list(range(ind + 1, ind + args.context_window + 1)))["translation"]]
            else:
                prev_samples, next_samples = [], []


            idioms_found = find_idioms([sample["translation_lemma"]], idioms)
            assert len(idioms_found) > 0
            translation_pairs[f'{args.dataset}_{args.lang}_{i}'] = {
                "text": sample["translation"]["en"],
                "original_text": sample["translation"][lang2],
                "contains_idioms": idioms_found[0],
                #"subtitle_id_files": sample["meta"]["subtitleId"] if args.dataset == "open_subtitles" else None,
                "context": {
                    "prev": prev_samples,
                    "next": next_samples
                }
            }
            i += 1
        output_file = f"data/external/{args.dataset}/{args.lang}_idioms"
        if args.index is not None:
            output_file += f"_{args.index}"
        if "filtered" in args.lang:
            output_file += "_filtered"
        if args.context_window is not None:
            ourput_file += "_context"

        print(f"Outputting idioms found to '{output_file}.json'")
        with open(f"{output_file}.json", "w") as f:
                json.dump(translation_pairs, f, ensure_ascii=False, indent=4, separators=(",", ": "))
