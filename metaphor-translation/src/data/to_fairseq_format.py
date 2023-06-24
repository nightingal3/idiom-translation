from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import stanza
import random
import pdb

# from this stackoverflow answer: https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append([x for x in range(i, i+len(pattern))])
    return matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert opensubtitles data to fairseq format")
    parser.add_argument("-l", "--lang", help="language to convert", choices=["fr", "fi", "jp"], required=True)
    parser.add_argument("--test_only", help="only do the test set (idioms)", action="store_true")
    parser.add_argument("--rand_test_only", help="only do the test set (random, from ted)", action="store_true")
    parser.add_argument("-d", "--duplicate", help="number of times to duplicate idiomatic phrases", type=int)
    parser.add_argument("--shuffle", help="shuffle lines", action="store_true")
    parser.add_argument('--test_within', action='store_true')
    parser.add_argument('--no-test_within', dest='test_within', action='store_false')
    parser.set_defaults(test_within=True)
    parser.add_argument("--holdout_idioms", help="Have a certain number of idioms in the held-out set (exclude all of these from the train or valid sets)", type=float, default=0)
    parser.add_argument("--per-word-labels", help="Also produce per-token labels for loss weighting", action="store_true")
    parser.add_argument("--tok_alignments_file", help="token alignments in Pharoah format (needed for word labels)", default="/compute/tir-0-15/alignments/jpen.awesome-align.out")
    parser.add_argument("--per-word-src_file", help="ordered src sequences aligned with tok alignments file (needed because of shuffle)", default="/compute/tir-0-15/mengyan3/data-fairseq/jp_no_test/train.jp")
    args = parser.parse_args()

    lang2 = "ja" if args.lang == "jp" else args.lang
    
    if args.per_word_src_file:
        with open(args.per_word_src_file, "r") as f:
            src = f.readlines()
            src = [x.strip() for x in src]
        tgt_file = args.per_word_src_file[:-2] + "en"
        with open(tgt_file, "r") as f:
            tgt = f.readlines()
            tgt = [x.strip() for x in tgt]

        if args.lang != "jp":
            per_word_src_file_valid = f"/compute/tir-0-15/mengyan3/data-fairseq/{args.lang}_perword_no_test/valid.{args.lang}"
            per_word_tgt_file_valid = f"/compute/tir-0-15/mengyan3/data-fairseq/{args.lang}_perword_no_test/valid.en"
        else:
            per_word_src_file_valid = f"/compute/tir-0-15/mengyan3/data-fairseq/{args.lang}_no_test/valid.jp"
            per_word_tgt_file_valid = f"/compute/tir-0-15/mengyan3/data-fairseq/{args.lang}_no_test/valid.en"

        with open(per_word_src_file_valid, "r") as f:
            src_valid = f.readlines()
            src_valid = [x.strip() for x in src_valid]
        with open(per_word_tgt_file_valid, "r") as f:
            tgt_valid = f.readlines()
            tgt_valid = [x.strip() for x in tgt_valid]

        dataset = []
        print("Constructing train dataset aligned with token alignments file")
        for i, j in tqdm(zip(src, tgt)):
            sample = {}
            sample["translation"] =  {"en": j, lang2: i}

            try:
                lemma = nlp(sample["translation"][lang2])
                sample["translation_lemma"] =  " ".join([word.lemma for sent in lemma.sentences for word in sent.words])
            except:
                sample["translation_lemma"] = sample["translation"][lang2]
            
            dataset.append(sample)
        
        print("Constructing valid dataset aligned with token alignments file")
        valid_dataset = []
        for i, j in zip(src_valid, tgt_valid):
            sample = {}
            sample["translation"] =  {"en": j, lang2: i}

            try:
                lemma = nlp(sample["translation"][lang2])
                sample["translation_lemma"] =  " ".join([word.lemma for sent in lemma.sentences for word in sent.words])
            except:
                sample["translation_lemma"] = sample["translation"][lang2]
            
            valid_dataset.append(sample)

    else:
        if not args.test_within:
            dataset = load_from_disk(f"./data/test_sets_final/{args.lang}/open_subtitles_{args.lang}-final.hf")
        else:
            dataset = load_from_disk(f"open_subtitles_{args.lang}-filtered_lemma_dataset.hf")

    idiom_dataset = pd.read_csv(f"./data/test_sets_final/{args.lang}/idiomatic_all.csv")
    literal_dataset = pd.read_csv(f"./data/test_sets_final/{args.lang}/literal.csv")
    random_dataset = pd.read_csv(f"./data/test_sets_final/{args.lang}/random_sents_ted.csv")

    idioms = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{args.lang}'")["idiom"].tolist()]
    
    if not args.test_only and not args.rand_test_only:

        if args.per_word_labels:
            sp = spm.SentencePieceProcessor(model_file="/projects/tir5/users/mengyan3/unilm/deltalm/spm.model")

        nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)
        lemmatized_idioms = [nlp(x) for x in idioms]
        idioms = set()
        for i, l in enumerate(lemmatized_idioms):
            sep = "" if args.lang == "jp" else " "
            lemma_idiom = sep.join([word.lemma for sent in l.sentences for word in sent.words])
            idioms.add(lemma_idiom)

        if isinstance(dataset, dict):
            dataset = dataset["train"]

        if not args.per_word_src_file:
            dataset_split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=False)

        if args.holdout_idioms > 0:
            # don't count sentences with multiple idioms as separate idiom types
            all_idioms = list(idiom_dataset["contains_idioms"].unique())
            all_idioms = [x.split(",") if "," in x else x for x in all_idioms]
            all_idioms = list(set([item.strip() if isinstance(sublist, list) else sublist.strip() for sublist in all_idioms for item in sublist]))
            all_idioms = sorted(all_idioms)

            train_idioms, test_idioms = train_test_split(all_idioms, test_size=args.holdout_idioms, random_state=42, shuffle=False)
            assert len(set(train_idioms) & set(test_idioms)) == 0
            
            with open(f"data-fairseq/train_idioms_{args.lang}.txt", "w") as f:
                for x in train_idioms:
                    f.write(f"{x}\n")
            with open(f"data-fairseq/test_idioms_{args.lang}.txt", "w") as f:
                for x in test_idioms:
                    f.write(f"{x}\n")
            
        base_dir = f"data-fairseq/{args.lang}"
        if args.duplicate is not None:
            base_dir += "_dup"
        if args.holdout_idioms > 0:
            base_dir += "_holdout"
        if args.per_word_labels:
            base_dir += "_perword"
        if not args.test_within:
            base_dir += "_no_test"

        train_src_name = f"{base_dir}/train.{args.lang}"
        train_tgt_name = f"{base_dir}/train.en"
        if args.duplicate is not None:
            train_src_name += f"_duplicated_{args.duplicate}"
            train_tgt_name += f"_duplicated_{args.duplicate}"

        Path.mkdir(Path(base_dir), parents=True, exist_ok=True)
        f_train_src = open(train_src_name, "w")
        f_train_tgt = open(train_tgt_name, "w")

        if not args.duplicate:
            f_train_lab = open(f"{base_dir}/train.label", "w")
            f_valid_lab = open(f"{base_dir}/valid.label", "w")
            if args.per_word_labels:
                f_train_lab_perword = open(f"{base_dir}/train.label_per_tok", "w")
                f_valid_lab_perword = open(f"{base_dir}/valid.label_per_tok", "w")
                
                alignments = []
                with open(args.tok_alignments_file) as f:
                    for line in f:
                        alignments.append(line.strip().split(" "))


        f_valid_src = open(f"{base_dir}/valid.{args.lang}", "w")
        f_valid_tgt = open(f"{base_dir}/valid.en", "w")
        
        src_lines, tgt_lines, src_labels, src_labels_per_word = [], [], [], []
        if not args.per_word_src_file:
            train_dataset = dataset_split["train"]
        else:
            train_dataset = dataset

        for ind, sample in enumerate(tqdm(train_dataset)):
            src_lines.append(f"{sample['translation'][lang2]}\n")
            tgt_lines.append(f"{sample['translation']['en']}\n")
            
            sample_text = sample["translation_lemma"] if args.lang != "jp" else "".join(sample["translation_lemma"].split())
            label = 1 if any(idiom in sample_text for idiom in idioms) else 0
            idioms_found = []
            for i in idioms:
                if i in sample_text:
                    idioms_found.append(i)
            
            if args.holdout_idioms > 0 and any(i in test_idioms for i in idioms_found):
                continue

            if args.duplicate is not None and label == 1:
                for _ in range(args.duplicate):
                    src_lines.append(f"{sample['translation'][lang2]}\n")
                    tgt_lines.append(f"{sample['translation']['en']}\n")
                    
            elif args.duplicate is None: #duplication and upweighting mutually exclusive
                src_labels.append(f"{label}\n")

                if args.per_word_labels:
                    curr_alignments = alignments[ind]
                    alignments_dict = {}
                    for align in curr_alignments:
                        try:
                            i, j = align.split("-")[0], align.split("-")[1]
                        except: # sometimes there are empty alignments
                            continue
                        alignments_dict[int(i)] = int(j)

                    tokenized_text = [x.replace("▁", "") for x in sp.encode(sample_text, out_type=str)]
                    tokenized_en_text = [x.replace("▁", "") for x in sp.encode(sample["translation"]["en"], out_type=str)]
                    if label == 1:
                        tokenized_idioms = [[x.replace("▁", "") for x in sp.encode(idiom, out_type=str)] for idiom in idioms_found]
                        tokenized_idioms = [[i for i in x if len(i) > 0] for x in tokenized_idioms]
                        label_list = ["0"] * len(tokenized_en_text)
                        all_idiom_indices = []
                        for idiom in tokenized_idioms:
                            idiom_indices = subfinder(tokenized_text, idiom)
                            all_idiom_indices.extend(idiom_indices)

                        all_idiom_indices_flat = set([item for sublist in all_idiom_indices for item in sublist])
                        try:
                            for i in all_idiom_indices_flat:
                                if i in alignments_dict:
                                    label_list[alignments_dict[i]] = "1"
                        except:
                            pdb.set_trace()
                        src_labels_per_word.append(" ".join(label_list))
                        f_train_lab_perword.write(" ".join(label_list) + "\n")
                    else:
                        zeros = ("0 " * len(tokenized_en_text))[:-1]
                        src_labels_per_word.append(zeros)
                        f_train_lab_perword.write(f"{zeros}\n")

        if args.duplicate is not None:
            train_data = list(zip(src_lines, tgt_lines))
            random.Random(42).shuffle(train_data) # shuffle the train data iff duplicated
        else:
            train_data = list(zip(src_lines, tgt_lines, src_labels))
        

        for s in train_data:
            if len(s) == 2:
                src_line, tgt_line = s
            else:
                src_line, tgt_line, src_lab = s
            f_train_src.write(src_line)
            f_train_tgt.write(tgt_line)
            if not args.duplicate:
                f_train_lab.write(src_lab)

        if not args.per_word_src_file:
            valid_dataset = dataset_split["test"]
        for sample in tqdm(valid_dataset):
            f_valid_src.write(f"{sample['translation'][lang2]}\n")
            f_valid_tgt.write(f"{sample['translation']['en']}\n")
            sample_text = sample["translation_lemma"] if args.lang != "jp" else "".join(sample["translation_lemma"].split())
            label = 1 if any(idiom in sample_text for idiom in idioms) else 0
    else:
        if args.test_only:
            Path.mkdir(Path(f"data-fairseq/{args.lang}-finalized_test_idioms"), parents=True, exist_ok=True)
            f_test_src = open(f"data-fairseq/{args.lang}-finalized_test_idioms/test.{args.lang}", "w")
            f_test_tgt = open(f"data-fairseq/{args.lang}-finalized_test_idioms/test.en", "w")
            Path.mkdir(Path(f"data-fairseq/{args.lang}-finalized_test_literal"), parents=True, exist_ok=True)
            f_literal_test_src = open(f"data-fairseq/{args.lang}-finalized_test_literal/test.{args.lang}", "w")
            f_literal_test_tgt = open(f"data-fairseq/{args.lang}-finalized_test_literal/test.en", "w")  
            
            for sample in tqdm(idiom_dataset.to_dict(orient="records")):
                f_test_src.write(f"{sample['original_text']}\n")
                f_test_tgt.write(f"{sample['text']}\n")
            
            for sample in tqdm(literal_dataset.to_dict(orient="records")):
                f_literal_test_src.write(f"{sample['original_text']}\n")
                f_literal_test_tgt.write(f"{sample['text']}\n")

        if args.rand_test_only:
            Path.mkdir(Path(f"data-fairseq/{args.lang}-finalized_test_rand"), parents=True, exist_ok=True)
            f_rand_test_src = open(f"data-fairseq/{args.lang}-finalized_test_rand/test.{args.lang}", "w")
            f_rand_test_tgt = open(f"data-fairseq/{args.lang}-finalized_test_rand/test.en", "w")
            for sample in tqdm(random_dataset.to_dict(orient="records")):
                f_rand_test_src.write(f"{sample['original_text']}\n")
                f_rand_test_tgt.write(f"{sample['text']}\n")

    if not args.test_only and not args.rand_test_only:
        print(f"data output to {base_dir}")
    elif args.test_only:
        print(f"data output to data-fairseq/{args.lang}-finalized_test_idiom and data-fairseq/{args.lang}-finalized_test_literal")
    elif args.rand_test_only:
        print(f"data output to data-fairseq/{args.lang}-finalized_test_rand")