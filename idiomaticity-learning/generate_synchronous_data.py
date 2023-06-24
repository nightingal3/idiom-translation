from typing import List, Callable, Tuple
from random import randint, choices
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import argparse
import pdb

#TODO: there is still a weird bug that occasionally causes
# a few extra (1-5) idioms to be generated with high numbers of sents.
# bother me but not worth looking into atm (just corrected the few cases manually)

def generate_symbols(max_symbols: int):
    for i in range(max_symbols):
        yield i, i + max_symbols


def generate_rule_set(depth: 1, num_rules: 10) -> List:
    # TODO: test depth 1 first, add variable depths later
    translations = []
    for d in range(depth):
        for t in generate_symbols(num_rules):
            translations.append(t)

    return translations

# note: include_rule means that EVERY sentence will have the idiom in it. 
# to generate a random set of sentences, set include_rule to None
def compose_random_sentences(
    rules: List,
    num_to_generate: int = 100,
    max_length: int = 6,
    include_rule: List = None, 
    add_context_beta: bool = False,
) -> Tuple:
    sents_src = []
    sents_trg = []
    num_generated = 0

    print("NUM TO GENERATE: ", num_to_generate)
    while num_generated < num_to_generate:
        curr_len = randint(1, max_length)
        rand_sent = choices(rules, k=curr_len)
        src_seq = [x[0] for x in rand_sent]
        trg_seq = [x[1] for x in rand_sent]

        if include_rule is not None:
            pattern, replacement = include_rule[0], include_rule[1]
            if curr_len < len(pattern):
                continue
            
            # only one idiom per sentence, depends on prev step though
            pattern_ind = find_sublist(pattern, src_seq)

            if len(pattern_ind) == 0:
                rule_insertion_ind = randint(0, curr_len - len(pattern))
                
                if add_context_beta:
                    src_seq.insert(max(0, rule_insertion_ind - 1), 11)
                    trg_seq.insert(max(0, rule_insertion_ind - 1), 21)
                    if rule_insertion_ind == 0:
                        rule_insertion_ind += 1

                src_seq[rule_insertion_ind : rule_insertion_ind + len(pattern)] = pattern
                trg_seq[rule_insertion_ind] = replacement
                del trg_seq[rule_insertion_ind + 1 : rule_insertion_ind + len(pattern)]

        num_generated += 1
        sents_src.append(src_seq)
        sents_trg.append(trg_seq)

    return sents_src, sents_trg


# Adapted from stackoverflow answer
# https://stackoverflow.com/questions/627435/how-to-remove-an-element-from-a-list-by-index
def find_sublist(sublist: List, lst: List):
    all_occurrences = []
    for ind in (i for i, e in enumerate(lst) if e == sublist[0]):
        if lst[ind : ind + len(sublist)] == sublist:
            all_occurrences.append((ind, ind + len(sublist) - 1))
    return all_occurrences

def replacement_rules(
    data_src: List,
    data_trg: List,
    orig_rules: List[List],
    replacement_rules: List[List],
    quota: int = 1000,
    quota_exact: bool = True,
    add_context_beta: bool = False,
):
    can_replace = []
    deleted_num = 0
    weird_times = 0
    for rule in replacement_rules:
        pattern, replacement = rule
        for i, s in enumerate(data_src):
            found_inds = find_sublist(pattern, s)

            if len(found_inds) == 0:
                can_replace.append(i)
            else:
                original_src = data_src[i][:]
                if len(found_inds) > 1: # have max one idiom per sentence
                    #pdb.set_trace()
                    for ind in found_inds[1:]:
                        del data_src[i][ind[0] : ind[1] + 1]
                        del data_trg[i][ind[0] : ind[1] + 1]
                        deleted_num += 1

                remaining_ind = found_inds[0]
                remaining_ind_1 = find_sublist(pattern, data_src[i])
                #if remaining_ind[0] != remaining_ind_1[0][0] or remaining_ind[1] != remaining_ind_1[0][1]:
                    #pdb.set_trace()
                if quota > 0:
                    if args.add_context_beta:
                        data_src[i].insert(
                            remaining_ind[0], 11
                        )  # make context "11" for now (unique)
                        data_trg[i].insert(remaining_ind[0], 21)
                        data_trg[i].insert(remaining_ind[0] + 1, replacement)
                        del data_trg[i][remaining_ind[0] + 2 : remaining_ind[1] + 3]
                    else:
                        data_trg[i][remaining_ind[0]] = replacement
                        del data_trg[i][remaining_ind[0] + 1 : remaining_ind[1] + 1]
                    quota -= 1
                else:
                    #pdb.set_trace()
                    if len(data_src[i]) <= 2: # just make some new random data
                        rand_len = randint(1, 5)
                        no_idioms_rules = [x for x in orig_rules if x[0] != 0 and x[0] != 1]
                        new_data = choices(no_idioms_rules, k=rand_len)
                        data_src[i] = [x[0] for x in new_data]
                        data_trg[i] = [x[1] for x in new_data]
                    else:
                        del data_src[i][remaining_ind[0] : remaining_ind[1] + 1]
                        del data_trg[i][remaining_ind[0] : remaining_ind[1] + 1]
                        if len(find_sublist(pattern, data_src[i])) > 0:
                            # patterns like 0 0 1 1 -> 0 1 cause this
                            new_ind = find_sublist(pattern, data_src[i])[0]
                            del data_src[i][new_ind[0] : new_ind[1] + 1]
                            del data_trg[i][new_ind[0] : new_ind[1] + 1]
                    deleted_num += 1

            
        print("quota left: ", quota)

        # TODO: this will break for multiple rules, should do all replacements after the loop
        if quota_exact and quota > 0: # we generated less than quota, so we need to add more
            new_src_sents, new_trg_sents = compose_random_sentences(
                orig_rules, include_rule=rule, num_to_generate=quota
            )
            to_replace = set(can_replace[: min(quota, len(can_replace) - 1)])
            print("Replacing sentences with idiomatic ones...")
            data_src = [x for i, x in enumerate(data_src) if i not in to_replace]
            data_trg = [x for i, x in enumerate(data_trg) if i not in to_replace]
            data_src.extend(new_src_sents)
            data_trg.extend(new_trg_sents)
        
    return data_src, data_trg


def output_sentences(
    src_sents: List,
    trg_sents: List,
    rules: List,
    output_dir: str,
    num_idioms: 1000,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    idiomatic_test_set: bool = True,
    add_context_beta: bool = False,
) -> None:
    # train_src, valid_src, train_trg, valid_trg = train_test_split(
    # src_sents, trg_sents, test_size=1000
    # )
    # Add non-compositional rule to test: 0 1 in src language should be 12 in trg instead of 10 11
    train_src, train_trg = replacement_rules(
        src_sents,
        trg_sents,
        rules,
        [[[0, 1], 12]],
        quota=num_idioms,
        quota_exact=True,
        add_context_beta=add_context_beta,
    )
    valid_src, valid_trg = compose_random_sentences(
        rules,
        num_to_generate=1000,
        include_rule=[[0, 1], 12],
        add_context_beta=add_context_beta,
    )

    if idiomatic_test_set:
        test_src, test_trg = compose_random_sentences(
            rules,
            num_to_generate=1000,
            include_rule=[[0, 1], 12],
            add_context_beta=add_context_beta,
        )

    print("Writing...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    write_lst_to_file(train_src, f"{output_dir}/train.input")
    write_lst_to_file(test_src, f"{output_dir}/test.input")
    write_lst_to_file(valid_src, f"{output_dir}/valid.input")
    write_lst_to_file(train_trg, f"{output_dir}/train.label")
    write_lst_to_file(test_trg, f"{output_dir}/test.label")
    write_lst_to_file(valid_trg, f"{output_dir}/valid.label")


def write_lst_to_file(lst: List, out_filename: str) -> None:
    with open(out_filename, "w") as tr_src:
        for line in lst:
            formatted_line = " ".join([str(x) for x in line])
            tr_src.write(f"{formatted_line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate synthetic data with a certain number of idiomatic phrases"
    )
    parser.add_argument(
        "-n",
        "--num_to_generate",
        help="size of train set",
        type=str,
        choices=["100", "1k", "10k", "100k", "1M", "10M"],
    )
    parser.add_argument(
        "-v",
        "--vocab_size",
        help="number of vocab tokens (same for both src and trg)",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--add_context_beta",
        help="try adding an informative context to the artificial idiom",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--num_idioms",
        help="number of idiom repetitions",
        type=str,
        choices=["1", "10", "100", "1k", "10k", "100k", "1M"],
    )
    args = parser.parse_args()

    num_map = {
        "1": 1,
        "10": 10,
        "100": 100,
        "1k": 1000,
        "10k": 10000,
        "100k": 100000,
        "1M": 1000000,
        "10M": 10000000,
    }

    num_to_generate = num_map[args.num_to_generate]
    num_idioms = num_map[args.num_idioms]

    print(f"Generating {num_to_generate} sentences with {num_idioms}")

    vocab = generate_rule_set(1, args.vocab_size)
    sents_src, sents_trg = compose_random_sentences(
        vocab, num_to_generate=num_to_generate
    )

    output_name = f"./data/corpus_{args.num_to_generate}_idioms_{args.num_idioms}"
    if args.add_context_beta:
        output_name += "_specialcontext"
    output_sentences(
        sents_src,
        sents_trg,
        vocab,
        output_name,
        num_idioms,
        add_context_beta=args.add_context_beta,
    )
