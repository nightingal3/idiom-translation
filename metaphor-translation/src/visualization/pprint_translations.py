import argparse
import pickle
import pandas as pd
from pprint import pformat
import json
import pdb

from src.utils.data_utils import find_idioms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="print translation file nicely.")
    parser.add_argument("-i", "--in_file", help="pickled translations file", required=True)
    parser.add_argument("-s", "--scores_file", help="score file (output from EaaS)")
    parser.add_argument("-o", "--out_file", help="file to write to", required=True)
    parser.add_argument("--csv", help="save translations to csv (to make annotation file for errors)", action="store_true")
    args = parser.parse_args()

    SELECT_N_FROM_IDIOMS = 5
    
    idioms = set([x.lower() for x in pd.read_csv("./data/external/all_idioms.csv")["idiom"].tolist()])
    out_filename = f"./data/processed/{args.in_file}.txt" if not args.out_file else args.out_file

    if args.scores_file:
        scores_dict = pickle.load(open(args.scores_file, "rb"))
        bleu_scores = scores_dict["scores"][0]["sample"]
        bert_scores = scores_dict["scores"][1]["sample"]
        rouge_scores = scores_dict["scores"][2]["sample"]

    if args.csv:
        with open(args.in_file, "rb") as in_f:
            if ".p" in args.in_file: 
                results = pickle.load(in_f)
                df = pd.DataFrame(results)
                df["references"] = df["references"].apply(lambda x: x[0])
            else:
                df = pd.read_json(args.in_file, orient="index")
                df = df[["contains_idioms","original_text", "text"]]
                
            # Should probably have kept track of this from the start...
            #idioms_found = find_idioms(df["source"], idioms)
            #df["contains_idioms"] = "
            df["trans_error"] = ""
            df["human_error"] = ""
            df["ambiguous"] = ""
            df["error_type"] = ""
            #df = df.groupby('contains_idioms').head(SELECT_N_FROM_IDIOMS)
            if "contains_idioms" in df.columns:
                df = df.sort_values(by="contains_idioms", axis=0)
            df.to_csv(out_filename, index=False)
    else:
        with open(args.in_file, "rb") as in_f:
            results = pickle.load(in_f)
            with open(out_filename, "w") as out_f:
                for i, r in enumerate(results):
                    s = pformat(r)
                    out_f.write(s)
                    if args.scores_file:
                        out_f.write("\n~ Scores ~\n")
                        out_f.write(f"BLEU: {bleu_scores[i]}\n")
                        out_f.write(f"BertScore: {bert_scores[i]}\n")
                        out_f.write(f"ROUGE: {rouge_scores[i]}\n")
                        out_f.write("\n===\n")
