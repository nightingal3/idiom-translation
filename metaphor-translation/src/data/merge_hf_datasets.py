import argparse
import os
from datasets import concatenate_datasets, load_from_disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="combine datasets specified by a language and directory")
    parser.add_argument("-d", "--dir", help="directory to look in", default="./")
    parser.add_argument("-l", "--lang", help="language code", choices=["fi", "fr", "jp"], default="fr")
    args = parser.parse_args()

    shards = []
    for filename in os.listdir(args.dir):
        if (filename.endswith(".hf") and args.lang in filename) and (f"{args.lang}-filtered") in filename: 
            print(filename)
            d = load_from_disk(filename)
            shards.append(d)
    
    dataset = concatenate_datasets(shards)
    dataset.save_to_disk(f"open_subtitles_{args.lang}-filtered_lemma_dataset.hf")