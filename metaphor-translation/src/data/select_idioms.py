import argparse
import pandas as pd
import pdb

from src.utils.data_utils import find_idioms, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="print translation file nicely.")
    parser.add_argument("-i", "--in_file", help="translations file (json)", required=True)
    parser.add_argument("-o", "--out_file", help="output file")
    args = parser.parse_args()

    SELECT_N_FROM_IDIOMS = 100

    if "jp-filtered" in args.in_file or "ja" in args.in_file:
        langname = "jp"
    elif "fi-filtered" in args.in_file:
        langname = "fi"
    elif "fr-filtered" in args.in_file:
        langname = "fr"
    else:
        langname = "unknown"

    json_dict = read_json(args.in_file)
    idioms = set([x.lower() for x in pd.read_csv("./data/external/all_idioms.csv")["idiom"].tolist()])

    contains_idioms_lst, en_lst, other_lst = [], [], []

    for sample in json_dict.values():
        contains_idioms_lst.append(sample["contains_idioms"])
        en_lst.append(sample["text"])
        other_lst.append(sample["original_text"])

    df = pd.DataFrame({"contains_idioms": contains_idioms_lst, "text": en_lst, "original_text": other_lst})
    df = df.drop_duplicates(subset="original_text")

    out_filename = f"./data/opensubtitles_final/{langname}_reduced" if not args.out_file else args.out_file
    #df = pd.read_csv(args.in_file)
    value_counts = df["contains_idioms"].value_counts()
    value_counts.to_csv(f"{out_filename}_value_counts.csv")

    df = df.groupby('contains_idioms').head(SELECT_N_FROM_IDIOMS)
    df = df.sort_values(by="contains_idioms", axis=0)
    df.to_csv(f"{out_filename}.csv", index=False)
    df.to_json(f"{out_filename}.json", orient="index", force_ascii=False)

    