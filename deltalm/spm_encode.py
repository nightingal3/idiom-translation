import sentencepiece as spm
import os
import argparse
import re
import pdb

def encode_spm(spm_model: str, text_file: str, out_dir: str) -> None:
    """
    """
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    with open(text_file, 'r') as f:
        lines = f.readlines()
    encodings = sp.encode(lines, out_type=str)
    encoded_lines = [' '.join(encoding) + '\n' for encoding in encodings]
    out_file = out_dir+ "/" +text_file.split("/")[-1]+ '.spm'
    with open(out_file, 'w') as f:
        f.writelines(encoded_lines)
    print("Written sentencepiece encoded text to", out_file, flush=True)
    return


if __name__=='__main__':
    """
    args:
        out_dir, phon_type, src1, tgt1, src2, tgt2, src1_lang, src2_lang,
        tgt_lang, train1_len, train2_len, val_len, test_len, config_temp
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
            help='Data file path',
            required=True)
    parser.add_argument('--out_dir', type=str,
              help='main dir where everything will be written',
              required=True)
    parser.add_argument('--model_name', type=str,
              help='Name of spm model',
              required=True)
    args = parser.parse_args()

    encode_spm(args.model_name, args.input, args.out_dir)
