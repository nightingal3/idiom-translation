from comet import download_model, load_from_checkpoint
from datasets import load_dataset
import argparse
from tqdm import tqdm
import pickle
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", help="language to use (opensubtitles)", choices=["fr", "ja", "fi"])
    args = parser.parse_args()

    model_path = download_model("wmt21-comet-qe-mqm")
    model = load_from_checkpoint(model_path)
    
    open_subtitles = load_dataset("open_subtitles", lang1="en", lang2=args.lang)["train"]
    data = []
    for sample in tqdm(open_subtitles):
        hyp = sample["translation"]["en"]
        src = sample["translation"][args.lang]
        data.append({"src": src, "mt": hyp, "id": sample["id"]})
    
    seg_scores, sys_score = model.predict(data, batch_size=16, gpus=1)
    all_scores_and_sents = []
    for score, sample in zip(seg_scores, data):
        all_scores_and_sents.append((score, sample["src"], sample["mt"], sample["id"]))

    all_scores_and_sents.sort(key=lambda x: x[0])
    with open(f"{args.lang}_cometqe.p", "wb") as f:
        pickle.dump(all_scores_and_sents, f)
