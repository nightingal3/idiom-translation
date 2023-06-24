import argparse
import json
import os
import pdb

from src.utils.data_utils import read_json, write_json, write_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="combine data from all datasets for each language")
    parser.add_argument("-d", "--dir",
                        help="directory to look in (should have subdirectories specifying names of datasets)",
                        default="./data/to_merge")
    parser.add_argument("-o", "--out", help="directory to output merged files to", default="./data/model_training")
    args = parser.parse_args()

    langs = {}
    for filename in os.listdir(args.dir):
        sub_path = os.path.join(args.dir, filename)
        print(sub_path)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if not f.endswith(".json") and not f.endswith(".jsonl"):
                    continue
                lang = f[:2]
                subfile_path = os.path.join(sub_path, f)
                data = read_json(subfile_path)
                data = {k: {**v, "source": sub_path} for k, v in data.items()}

                if lang not in langs:
                    langs[lang] = list(data.values())
                else:
                    langs[lang].extend(list(data.values()))

    for lang in langs: # write to json
        #data_dict = {i: d for i, d in enumerate(langs[lang])}
        write_json(langs[lang], f"{args.out}/{lang}")
        write_parallel(langs[lang], f"{args.out}/parallel/{lang}/all.{lang}", f"{args.out}/parallel/en/{lang}.en")

    print(f"Written to {args.out}/{lang}.json and {args.out}/parallel/")




