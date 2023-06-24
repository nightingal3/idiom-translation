import pandas as pd
import argparse
from datasets import Dataset, load_from_disk, load_dataset
from tqdm import tqdm
import stanza
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert opensubtitles data to fairseq format")
    parser.add_argument("-l", "--lang", help="language to convert", choices=["fr", "fi", "jp"], required=True)
    args = parser.parse_args()

    lang2 = "ja" if args.lang == "jp" else args.lang
    nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)

    dataset = load_from_disk(f"open_subtitles_{args.lang}-filtered_lemma_dataset.hf")
    idiom_dataset = pd.read_csv(f"./data/opensubtitles_final/{args.lang}_reduced.csv")
    if "train" in dataset.keys():
        dataset = dataset["train"]
    idioms = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{args.lang}'")["idiom"].tolist()]
    test_set_idiomatic = pd.read_csv(f"./data/test_sets_final/{args.lang}/idiomatic_all.csv")
    test_set_literal = pd.read_csv(f"./data/test_sets_final/{args.lang}/literal.csv")
    test_set_all = pd.concat([test_set_idiomatic, test_set_literal])
    test_set_all["lemma"] = test_set_all["original_text"].apply(lambda x: " ".join([y.lemma for y in nlp(x).sentences[0].words]))
    test_phrases = set(test_set_all["original_text"])

    train_phrases = []
    num_skipped = 0
    for i, sample in enumerate(tqdm(dataset)):
        if sample["translation"][lang2] in test_phrases: 
            num_skipped += 1
            continue
        else:
            train_phrases.append(sample)

    new_train_df = pd.DataFrame(train_phrases)
    new_dataset = Dataset.from_pandas(new_train_df)
    new_dataset.save_to_disk(f"./data/test_sets_final/{args.lang}/open_subtitles_{args.lang}-final.hf")