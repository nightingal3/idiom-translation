import datasets
import pandas as pd
import numpy as np
import string
import re
import pdb

def count_len_in_words(sentence):
    return len(re.sub(r'[^\w\s]','', sentence).split(" "))

def count_len_in_words_hf(example):
    no_punct = re.sub(r'[^\w\s]','', example["translation"]["en"]).strip()
    example["word_len"] = len(no_punct.split(" "))
    return example

def extract_sents_to_format(example):
    langs = list(example["translation"].keys())
    example["text"] = example["translation"]["en"]
    langs.remove("en")
    example["original_text"] = example["translation"][langs[0]]
    return example

def get_one_lang_ted(example):
    lang = example["lang"]
    idx = example["translations"]["language"].index(lang)
    en_idx = example["translations"]["language"].index("en")
    example["translation"] = {lang: example["translations"]["translation"][idx], "en": example["translations"]["translation"][en_idx]}
    return example

def get_closest_idx(small_lst, big_lst):
    small_lst = np.array(small_lst)
    big_lst = np.array(big_lst)
    closest_inds = []
    # find closest without replacements, needs loop
    for i in range(len(small_lst)):
        tmp = abs(small_lst[i] - big_lst)
        idx = np.argmin(tmp)
        closest_inds.append(idx)
        # put a big num in the place of the found index so that it is not found again
        big_lst[idx] = 9999
    
    return closest_inds

def load_large_and_small_dataset(lang, large_dataset="ted_talks"):
    if large_dataset == "open_subtitles":
        dataset_large = datasets.load_from_disk(f"./data/test_sets_final/{lang}/open_subtitles_{lang}-final.hf").shuffle(seed=42).shard(index=0, num_shards=100)
        dataset_large = dataset_large.map(count_len_in_words_hf)
    else: # use ted talks dataset
        lang2 = lang if lang != "jp" else "ja"
        dataset_large = datasets.load_dataset("ted_multi")["train"]
        dataset_large = dataset_large.filter(lambda example: lang2 in example["translations"]["language"] and "en" in example["translations"]["language"])
        dataset_large = dataset_large.add_column("lang", [lang2] * len(dataset_large))
        dataset_large = dataset_large.map(get_one_lang_ted)
        dataset_large = dataset_large.map(count_len_in_words_hf)

    dataset_idiom = pd.read_csv(f"./data/test_sets_final/{lang}/idiomatic_all.csv")
    dataset_literal = pd.read_csv(f"./data/test_sets_final/{lang}/literal.csv")
    dataset_small = pd.concat([dataset_idiom, dataset_literal], ignore_index=True)
    dataset_small["word_len"] = dataset_small["text"].apply(count_len_in_words)
    dataset_small = dataset_small.sort_values(by="word_len", ascending=False)

    return dataset_large, dataset_small

if __name__ == "__main__":
    langs = ["fi", "fr", "jp"]
    dataset = "ted_talks"
    for lang in langs:
        dataset_large, dataset_small = load_large_and_small_dataset(lang, large_dataset=dataset)
        closest_rand_sents = dataset_large[get_closest_idx(dataset_small["word_len"].tolist(), dataset_large["word_len"])]
        rand_dataset = datasets.Dataset.from_dict(closest_rand_sents)
        rand_dataset = rand_dataset.map(extract_sents_to_format)
        if dataset == "open_subtitles":
            rand_dataset = rand_dataset.remove_columns(["translation", "word_len", "translation_lemma"])
        else:
            rand_dataset = rand_dataset.remove_columns(["translation", "word_len", "lang", "talk_name", "translations"])
        rand_dataset.to_csv(f"./data/test_sets_final/{lang}/random_sents_ted.csv", index=False)
    
