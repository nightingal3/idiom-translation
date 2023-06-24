import argparse
import torch
from transformers import EncoderDecoderModel, BertTokenizer, PreTrainedModel, logging
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data.dictionary import Dictionary

import re
import os
import json
import pdb

# Parts of this script were inspired by/lifted from this post:
# https://huggingface.co/blog/porting-fsmt

JSON_INDENT = 2

class DeltaLMModel(PreTrainedModel):
    def __init__(self, state, config):
        super().__init__(config)
        self.config = config
        self.encoder = DeltaLMEncoder(config)
        self.decoder = DeltaLMDecoder(config)

def rewrite_dict_keys(d):
    # (1) remove word breaking symbol, (2) add word ending symbol where the word is not broken up,
    # e.g.: d = {'le@@': 5, 'tt@@': 6, 'er': 7} => {'le': 5, 'tt': 6, 'er</w>': 7}
    d2 = dict((re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v) for k, v in d.items())
    keep_keys = "<s> <pad> </s> <unk>".split()
    # restore the special tokens
    for k in keep_keys:
        del d2[f"{k}</w>"]
        d2[k] = d[k]  # restore
    return d2

def convert_deltalm_checkpoint_from_disk(checkpoint_path: str, output_path: str, dict_path: str = "dict.txt"):
    """Convert a DELTALM checkpoint from disk to HF format."""
    # load the DELTALM checkpoint
    # TODO: it doesn't allow loading the pretrained checkpoint...
    assert os.path.exists(checkpoint_path)
    os.makedirs(output_path, exist_ok=True)
    print(f"Loading deltalm checkpoint from {checkpoint_path}")

    checkpoint_file = os.path.basename(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)

    # load the src/tgt dict (they're the same for this model)
    src_dict = Dictionary.load(os.path.join(os.getcwd(), dict_path))
    tgt_dict = src_dict
    src_vocab = rewrite_dict_keys(src_dict.indices)
    tgt_vocab = src_vocab
    pdb.set_trace()
    
    # load the config
    config = state["cfg"]
    model_cfg = vars(config["model"])
    model_config_output = os.path.join(output_path, "config.json")
    # TODO: tokenizer/bpe in config seem to be None
    model_conf = { # TODO: may have to change this depending on my deltalm implementation
        "architectures": ["EncoderDecoderModel"],
        "model_type": "DeltaLM",
        "dropout": model_cfg["dropout"],
        "attention_dropout": model_cfg["attention_dropout"],
        "encoder_attention_heads": model_cfg["encoder_attention_heads"],
        "encoder_ffn_embed_dim": model_cfg["encoder_ffn_embed_dim"],
        "encoder_layers": model_cfg["encoder_layers"],
        "encoder_layerdrop": model_cfg["encoder_layerdrop"],
        "decoder_attention_heads": model_cfg["decoder_attention_heads"],
        "decoder_ffn_embed_dim": model_cfg["decoder_ffn_embed_dim"],
        "decoder_layers": model_cfg["decoder_layers"],
        "decoder_layerdrop": model_cfg["decoder_layerdrop"],
        "max_position_embeddings": model_cfg["max_positions"],
        "scale_embedding": not model_cfg["no_scale_embedding"],
        "tie_word_embeddings": model_cfg["share_decoder_input_output_embed"],
        "bos_token_id": tgt_dict.bos(),
        "pad_token_id": tgt_dict.pad(),
        "eos_token_id": tgt_dict.eos(),
        "is_encoder_decoder": True,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
    }

    # TODO: check if this is what fairseq is doing
    model_conf["num_beams"] = 5
    model_conf["early_stopping"] = False
    model_conf["length_penalty"] = 1.0

    print(f"Generating {model_config_output}")
    with open(model_config_output, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=JSON_INDENT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to DELTALM checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output HF checkpoint")
    parser.add_argument("--dict_path", type=str, default="dict.txt", help="Path to src/tgt dictionary")
    args = parser.parse_args()
    convert_deltalm_checkpoint_from_disk(args.checkpoint_path, args.output_path, dict_path=args.dict_path)