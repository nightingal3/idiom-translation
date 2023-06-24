import torch
from typing import List, Union
import io
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from timeit import default_timer as timer
from tqdm import tqdm
import argparse
import pdb
import sys

from transformer_model import Seq2SeqTransformer
from generate_synchronous_data import find_sublist

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
SRC_LANGUAGE = "src"
TGT_LANGUAGE = "trg"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
SCRATCH_DIR = "/scratch/mengyan3"

# Place-holders
token_transform = {}
vocab_transform = {}

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def setup_vocab():
    token_transform[SRC_LANGUAGE] = get_tokenizer(None)
    token_transform[TGT_LANGUAGE] = get_tokenizer(None)

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(
            yield_tokens(simulated_data["train"][ln]),
            min_freq=1,
            specials=special_symbols,
            special_first=True,
        )

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)


def get_lang_iterator(split: str = "train"):
    with open(simulated_data[split][SRC_LANGUAGE], "r") as f:
        src_lines = f.read().splitlines()
    with open(simulated_data[split][TGT_LANGUAGE], "r") as f:
        trg_lines = f.read().splitlines()

    return list(zip(src_lines, trg_lines))


def yield_tokens(file_path):
    with io.open(file_path) as f:
        for line in f:
            yield line.strip().split()


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = get_lang_iterator()
    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    
    pbar = tqdm(total=len(train_dataloader))

    for i, (src, tgt) in enumerate(train_dataloader):
        if i != 0 and i % 1000 == 0:
            pbar.update(1000)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # convert tgt to long
        tgt = tgt.to(torch.long).to(DEVICE)
        assert tgt.dtype == torch.long
        
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        #loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1).to(torch.long))
        try:
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        except:
            pdb.set_trace()
        loss.backward()

        optimizer.step()
        losses += loss.item()

    pbar.close()
    
    return losses / len(train_dataloader)


def evaluate(model, split: str = "valid"):
    model.eval()
    losses = 0

    val_iter = get_lang_iterator(split=split)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm(val_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # seems like an issue with some of the files/happens intermittently
        tgt = tgt.type(torch.long).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses += loss.item()

    return losses / len(val_dataloader)


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return (
        " ".join(
            vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))
        )
        .replace("<bos>", "")
        .replace("<eos>", "")
    )


def evaluate_idiomatic_translation(model):
    model.eval()
    correct = 0

    test_iter = get_lang_iterator(split="test")
    test_dataloader = DataLoader(
        test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    for src, tgt in tqdm(test_iter):
        src_seq = [int(x) for x in src.split()]
        # there's always exactly one idiom in the test sequence
        idiom_index = find_sublist([0, 1], src_seq)[0]
        tgt_translation = [int(x) for x in translate(model, src).split()]
        if (
            len(tgt_translation) < len(src_seq) - 2
            or idiom_index[0] > len(tgt_translation) - 1
        ):
            continue
        try:
            if tgt_translation[idiom_index[0]] == 12:
                correct += 1
        except:
            pdb.set_trace()

    return correct / len(test_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train a seq2seq transformer on simulated data"
    )
    parser.add_argument(
        "-t",
        "--transformer_size",
        help="transformer size (see code for s/m/l settings)",
        default="s",
        choices=["s", "m", "l"],
    )
    parser.add_argument(
        "-c",
        "--corpus_size",
        help="Total corpus size (train)",
        required=True,
        type=str,
        choices=["100k", "1M", "10M"],
    )
    parser.add_argument(
        "-i",
        "--idiom_size",
        help="Number of idiom repetitions",
        required=True,
        type=str,
        choices=["10", "100", "1k", "10k", "100k", "1M"],
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="random seed to use",
        type=int,
        default=42
    )
    parser.add_argument(
        "-e",
        "--early_stopping",
        help="Use early stopping",
        action="store_true"
    )
    parser.add_argument(
        "--use_context_beta", help="Try informative context", action="store_true"
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    print("running with args: ", args)

    dataset_name = f"corpus_{args.corpus_size}_idioms_{args.idiom_size}"
    if args.use_context_beta:
        dataset_name += "_specialcontext"

    simulated_data = {
        "train": {
            SRC_LANGUAGE: f"{SCRATCH_DIR}/data/{dataset_name}/train.input",
            TGT_LANGUAGE: f"{SCRATCH_DIR}/data/{dataset_name}/train.label",
        },
        "valid": {
            SRC_LANGUAGE: f"{SCRATCH_DIR}/data/{dataset_name}/valid.input",
            TGT_LANGUAGE: f"{SCRATCH_DIR}/data/{dataset_name}/valid.label",
        },
        "test": {
            SRC_LANGUAGE: f"{SCRATCH_DIR}/data/{dataset_name}/test.input",
            TGT_LANGUAGE: f"{SCRATCH_DIR}/data/{dataset_name}/test.label",
        },
    }
    setup_vocab()

    transformer_sizes = {
        "s": {
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "hid_dim": 512,
            "emb_size": 512,
            "nheads": 16,
            "num_epochs": 15 if args.use_context_beta else 10
        },
        "m": {
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "hid_dim": 512,
            "emb_size": 512,
            "nheads": 16,
            "num_epochs": 15 if args.use_context_beta else 20
        },
        "l": {
            "num_encoder_layers": 16,
            "num_decoder_layers": 16,
            "hid_dim": 512,
            "emb_size": 512,
            "nheads": 16,
            "num_epochs": 25 if args.use_context_beta else 30
        },
    }

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = transformer_sizes[args.transformer_size]["emb_size"]
    NHEAD = transformer_sizes[args.transformer_size]["nheads"]
    FFN_HID_DIM = transformer_sizes[args.transformer_size]["hid_dim"]
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = transformer_sizes[args.transformer_size]["num_encoder_layers"]
    NUM_DECODER_LAYERS = transformer_sizes[args.transformer_size]["num_decoder_layers"]

    transformer = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],  # Tokenization
            vocab_transform[ln],  # Numericalization
            tensor_transform,
        )  # Add BOS/EOS and create tensor

    NUM_EPOCHS = transformer_sizes[args.transformer_size]["num_epochs"] if not args.early_stopping else 1000
    print(f"=== Training for {NUM_EPOCHS} epochs ===")

    patience = 10
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch} (patience {patience})")
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print(f"Epoch {epoch} (val loss {val_loss})")
        if args.early_stopping:
            if val_loss < best_val_loss:
                print(f"New best val loss: {val_loss} (was {best_val_loss})")
                best_val_loss = val_loss
                patience = 10
                torch.save(transformer.state_dict(), f"{SCRATCH_DIR}/models/{dataset_name}.pt")
            else:
                patience -= 1
                if patience == 0:
                    break

    print(f"PARAMS: corpus size {args.corpus_size}, idioms {args.idiom_size}, transformer {args.transformer_size} with context? {args.use_context_beta}")
    test_acc = evaluate_idiomatic_translation(transformer)
    print(test_acc)
