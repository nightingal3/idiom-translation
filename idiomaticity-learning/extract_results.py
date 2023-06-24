import argparse
import pandas as pd
import os
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_file", type=str, required=True, help="Path to experiment file used to run experiments (tsv format)")
    parser.add_argument("-s", "--slurm_id", type=int, help="Slurm ID of experiment (will be inferred otherwise)")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to experimental results files")
    parser.add_argument("-o", "--output_path", type=str, help="Output file name")
    parser.add_argument("-f", "--format_file", type=str, help="Path to format file that specifies table layout (csv format)")
    parser.add_argument("-a", "--average_cols", nargs="+", help="Columns to average over")
    args = parser.parse_args()

    exp_df = pd.read_csv(args.experiment_file, sep="\t")
    exp_results = ["incomplete"] * len(exp_df)

    # TODO: take majority vote of slurm files in curr dir as slurm id if not specified
    slurm_prefix = f"{args.input_dir}/slurm-{args.slurm_id}_"

    for i, row in exp_df.iterrows():
        if os.path.exists(f"{slurm_prefix}{row['TaskID']}.out"):
            # read last line of output file
            with open(f"{slurm_prefix}{row['TaskID']}.out", "r") as f:
                last_line = f.readlines()[-1]
            try:
                last_line = float(last_line.strip())
                exp_results[i] = last_line
                print(f"Task {row['TaskID']} completed with result: {last_line}")

            except:
                print(f"Task {row['TaskID']} is still running")
                continue

    exp_df["Results"] = exp_results
    # remove the command column
    exp_df = exp_df.drop(columns=["Command"])
    exp_settings = ["CorpusSize", "IdiomSize", "TransformerSize"]
    if args.average_cols:
        if "incomplete" in exp_results:
            raise ValueError("Cannot average over incomplete experiments")

        grouped_df = exp_df.groupby(exp_settings)
        avg_df = grouped_df.mean().reset_index()
        # group by all other columns except for the average columns
        exp_df = avg_df
        exp_df = exp_df.drop(columns=args.average_cols)

    output_path = args.output_path if args.output_path else f"results.tsv"
    exp_df.to_csv(output_path, index=False, sep="\t")



