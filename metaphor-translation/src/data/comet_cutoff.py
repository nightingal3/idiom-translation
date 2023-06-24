import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import seaborn as sns
import pdb
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="pickled input file with scores", type=str, required=True)
    parser.add_argument("-t", "--threshold", help="threshold to cut off (lines will be cut if they are in the bottom x%)", type=int, default=10)
    args = parser.parse_args()

    all_lines = pickle.load(open(args.filename, "rb"))
    threshold_ind = int((len(all_lines) / 100) * args.threshold)
    cutoff_val = all_lines[threshold_ind][0]
    print("cutoff comet-qe score: ", cutoff_val)
    all_lines = all_lines[threshold_ind:]
    df = pd.DataFrame(all_lines, columns=["score", "other-lang", "eng", "index"])
    sns.histplot(data=df, x="score", kde=True)

    filename_no_ext = args.filename[:-2]
    df.to_csv(f"./data/processed/{filename_no_ext}_comet_filtered.csv", index=False)
    plt.savefig(f"./{filename_no_ext}_scores_after.png")

    translation_pairs = []
    for i, (score, jp_s, en_s) in enumerate(all_lines):
        translation_pairs.append({
                "translation": {
                    "en": en_s,
                    "fr": jp_s
                }
            })
    with open(f"data/processed/{filename_no_ext}_comet_filtered.json", "w") as f:
        for pair in translation_pairs:
            json.dump(pair, f, ensure_ascii=False, separators=(",", ": "))
            f.write("\n")