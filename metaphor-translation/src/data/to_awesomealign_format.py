import sentencepiece as spm
import argparse
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert two parallel data files to format for awesomealign")
    parser.add_argument("-l", "--lang", help="language", choices=["fr", "fi", "jp"], default="jp")
    parser.add_argument("-s", "--source", help="source file", default="/compute/tir-0-15/mengyan3/data-fairseq/jp_no_test/train.jp")
    parser.add_argument("-t", "--target", help="target file", default="/compute/tir-0-15/mengyan3/data-fairseq/jp_no_test/train.en")
    parser.add_argument("-o", "--output", help="output file", default="/compute/tir-0-15/alignments/jp/jpen.src-tgt")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file="/projects/tir5/users/mengyan3/unilm/deltalm/spm.model")
    with open(args.source, "r") as f:
        src = f.readlines()
        src_tokenized = [sp.encode(x.strip(), out_type=str) for x in src]
        src_tokenized = [" ".join(x).replace('▁', ' ').strip() for x in src_tokenized]
        

    with open(args.target, "r") as f:
        tgt = f.readlines()
        tgt_tokenized = [sp.encode(x.strip(), out_type=str) for x in tgt]
        tgt_tokenized = [" ".join(x).replace('▁', ' ').strip() for x in tgt_tokenized]

    with open(args.output, "w") as f:
        for i, j in zip(src_tokenized, tgt_tokenized):
            f.write(f"{i} ||| {j}\n")



