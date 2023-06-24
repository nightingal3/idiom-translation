import pandas as pd
import pdb
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input directory")
    parser.add_argument("-o", "--output", help="output file")
    args = parser.parse_args()

    src_file = os.path.join(args.input, "srcs.txt")
    tgt_file = os.path.join(args.input, "tgts.txt")
    hyp_file = os.path.join(args.input, "hyps.txt")
    with open(src_file, "r") as src_f:
        srcs = [l.strip() for l in src_f.readlines()]
    with open(tgt_file, "r") as tgt_f:
        tgts = [l.strip() for l in tgt_f.readlines()]
    with open(hyp_file, "r") as hyp_f:
        hyps = [l.strip() for l in hyp_f.readlines()]

    src_df = pd.DataFrame(srcs)
    tgt_df = pd.DataFrame(tgts)
    hyp_df = pd.DataFrame(hyps)
    df = pd.concat([src_df, tgt_df, hyp_df], axis=1)
    df.columns = ["src", "tgt", "hyp"]
    df.to_csv(args.output, index=False)