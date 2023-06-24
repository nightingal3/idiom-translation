import argparse
import random
import numpy as np
from tqdm import tqdm
from evaluate_translations import automatic_metrics
import evaluate

def bootstrap_resampling(srcs, refs, sys1, sys2, num_iterations=1000):
    assert len(refs) == len(sys1) == len(sys2), "All inputs must have the same length"
    
    # Calculate scores
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")

    _, BLEU_sys1, _, _, _ = automatic_metrics(srcs, sys1, refs, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
    _, BLEU_sys2, _, _, _ = automatic_metrics(srcs, sys2, refs, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
    # Compute observed difference
    obs_diff = BLEU_sys1["bleu"] - BLEU_sys2["bleu"]
    
    count = 0
    for _ in range(num_iterations):
        # Generate bootstrap samples
        rand_inds = np.random.randint(0, len(sys1), len(sys1))
        sys1_rand = [sys1[i] for i in rand_inds]
        sys2_rand = [sys2[i] for i in rand_inds]
        _, bootstrap_sys1, _, _, _ = automatic_metrics(srcs, sys1_rand, refs, silent=True, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
        _, bootstrap_sys2, _, _, _ = automatic_metrics(srcs, sys2_rand, refs, silent=True, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
        # Compute difference for bootstrap samples
        bootstrap_diff = bootstrap_sys1["bleu"] - bootstrap_sys2["bleu"]
        
        # Count if bootstrap difference is greater than or equal to observed difference
        if bootstrap_diff >= obs_diff:
            count += 1
    
    # Compute p-value
    p_value = count / num_iterations
    
    return p_value


def permutation_test(refs, sys1, sys2, num_iterations=1000):
    assert len(refs) == len(sys1) == len(sys2), "All inputs must have the same length"
    
    # Calculate BLEU scores
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")

    _, BLEU_sys1, _, _, _ = automatic_metrics(srcs, sys1, refs, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
    _, BLEU_sys2, _, _, _ = automatic_metrics(srcs, sys2, refs, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)

    # Compute observed difference
    obs_diff = BLEU_sys2["bleu"] - BLEU_sys1["bleu"]
    
    count = 0
    for _ in tqdm(range(num_iterations)):
        new_sys1, new_sys2 = [], []
        for ref, hyp1, hyp2 in zip(refs, sys1, sys2):
            if random.random() < 0.5:
                new_sys1.append(hyp1)
                new_sys2.append(hyp2)
            else:
                new_sys1.append(hyp2)
                new_sys2.append(hyp1)

        # Calculate new BLEU scores
        _, new_scores_sys1, _, _, _ = automatic_metrics(srcs, new_sys1, refs, silent=True, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
        _, new_scores_sys2, _, _, _ = automatic_metrics(srcs, new_sys2, refs, silent=True, bleu=bleu, rouge=rouge, bertscore=bertscore, meteor=meteor)
        # Compute difference for new test set
        new_diff = new_scores_sys2["bleu"] - new_scores_sys1["bleu"]

        # Count if new difference is greater than or equal to observed difference
        if new_diff >= obs_diff:
            print("higher than observed")
            count += 1
    
    # Compute p-value
    p_value = count / num_iterations
    
    return p_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i1",
        "--input1",
        help="input1 dirname",
    )
    parser.add_argument(
        "-i2",
        "--input2",
        help="input2 dirname",
    )
    parser.add_argument("--test_type", type=str, default="bootstrap", choices=["bootstrap", "permutation"])
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    refs = [line.rstrip() for line in open(f"{args.input1}/tgts.txt", "r")]
    srcs = [line.rstrip() for line in open(f"{args.input1}/srcs.txt", "r")]
    hyps1 = [line.rstrip() for line in open(f"{args.input1}/hyps.txt", "r")]
    if "\t" in hyps1[0]: # sometimes the score gets left in...
        hyps1 = [x.split("\t")[1] for x in hyps1]
    
    refs2 = [line.rstrip() for line in open(f"{args.input2}/tgts.txt", "r")]
    srcs2 = [line.rstrip() for line in open(f"{args.input2}/srcs.txt", "r")]
    hyps2 = [line.rstrip() for line in open(f"{args.input2}/hyps.txt", "r")]
    if "\t" in hyps2[0]: # sometimes the score gets left in...
        hyps2 = [x.split("\t")[1] for x in hyps2]

    if refs != refs2 or srcs != srcs2:
        raise ValueError("References or sources don't match")

    if args.test_type == "bootstrap":
        p_val = bootstrap_resampling(srcs, refs, hyps1, hyps2, num_iterations=args.num_samples)
    elif args.test_type == "permutation":
        p_val = permutation_test(refs, hyps1, hyps2, num_iterations=args.num_samples)

    print(f"=== p-value: {p_val} with {args.num_samples} iterations ===")