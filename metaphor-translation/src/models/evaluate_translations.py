import evaluate
import pickle
import pandas as pd
import argparse
import stanza
import ast
import os
from inspiredco.critique import Critique
import time
import pdb

client = Critique(os.environ.get("INSPIREDCO_API_KEY"))

def get_paraphrase_rate(
    df: pd.DataFrame,
    idiom_df: pd.DataFrame,
    stanza_pipeline: stanza.Pipeline,
    lang: str,
    hyps=[],
    refs=[],
    **other_metrics,
):
    idioms = list(idiom_df["idiom"])
    lemmatized_idioms = [stanza_pipeline(x) for x in idioms]
    idioms = []
    for i, l in enumerate(lemmatized_idioms):
        sep = "" if lang == "jp" else " "
        lemma_idiom = sep.join(
            [word.lemma for sent in l.sentences for word in sent.words]
        )
        idioms.append(lemma_idiom)

    idiom_df["idiom_lemma"] = idioms
    idioms = set(idioms)
    literalized_idioms = 0
    sents_to_count = 0
    for sample in df.to_dict(orient="records"):
        sent = sample["sent"]
        src_doc = stanza_pipeline(sent)
        lemma_src_lst = [
            word.lemma for sent in src_doc.sentences for word in sent.words
        ]
        if lang == "jp":
            lemma_src = "".join(lemma_src_lst)
        else:
            lemma_src = " ".join(lemma_src_lst)
        idioms_found = []
        for i in idioms:
            if i in lemma_src:
                idioms_found.append(i)

        at_least_one_literal = False
        for idiom in idioms_found:
            literal_translations = idiom_df.query(f'idiom_lemma == "{idiom}"')
            try:
                noun_translation = [
                    x.lower()
                    for x in ast.literal_eval(
                        literal_translations["noun_translation"][0].item()
                    )
                ]
            except:  # if there are duplicate idioms after lemmatization, just take the first one
                noun_translation = [
                    x.lower()
                    for x in ast.literal_eval(
                        literal_translations["noun_translation"].iloc[0]
                    )
                ]

            if any(w in sample["translation"] for w in noun_translation):
                at_least_one_literal = True

        if at_least_one_literal:
            literalized_idioms += 1

    print("PARAPHRASE RATE:")
    print(literalized_idioms, len(df), 1 - (literalized_idioms / len(df)))

    other_results = []
    for metric in other_metrics:
        print(metric)
        metric_fn = other_metrics[metric]
        results = metric_fn.compute(predictions=hyps, references=refs)
        other_results.append(results)
        print(results)

    return 1 - (literalized_idioms / len(df)), other_results


def automatic_metrics(srcs, hyps, refs, heuristic: bool = False, silent: bool = False, bleu: evaluate.Evaluator = None, rouge: evaluate.Evaluator = None, bertscore: evaluate.Evaluator = None, meteor: evaluate.Evaluator = None):
    if bleu is None or rouge is None or bertscore is None or meteor is None:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")
        meteor = evaluate.load("meteor")

    s0 = time.perf_counter()
    results_bleu = bleu.compute(predictions=hyps, references=refs)
    s1 = time.perf_counter()
    
    results_rouge = rouge.compute(predictions=hyps, references=refs)
    results_bertscore = bertscore.compute(predictions=hyps, references=refs, lang="en")
    results_meteor = meteor.compute(predictions=hyps, references=refs)

    if not silent:
        print("TIME FOR BLEU:", s1 - s0)
        print("BLEU\n", results_bleu)
        print("ROUGE\n", results_rouge)
        print("BERTSCORE\n", sum(results_bertscore["f1"]) / len(results_bertscore["f1"]))
        print("METEOR\n", results_meteor)

    df = pd.DataFrame({"sent": srcs, "reference": refs, "translation": hyps})
    if heuristic:
        idiom_df = pd.read_csv(
            "./data/external/all_idioms_naive_translation.csv"
        ).query(f"lang == '{args.lang}'")
        lang2 = "ja" if args.lang == "jp" else args.lang
        nlp = stanza.Pipeline(
            lang=lang2, processors="tokenize,pos,lemma", pos_batch_size=1000
        )

        paraphrase_rate = get_paraphrase_rate(df, idiom_df, nlp, args.lang, hyps, refs, bleu=bleu)

        return results_rouge, results_bertscore, results_meteor, paraphrase_rate

    return df, results_bleu, results_rouge, results_bertscore, results_meteor

def metrics_with_critique(srcs, hyps, refs, heuristic=False):
    dataset = list({"source": src, "target": hyp, "references": [ref]} for src, hyp, ref in zip(srcs, hyps, refs))
    s0 = time.perf_counter()
    result = client.evaluate(
        metric="bleu",
        dataset=dataset,
        config={}
    )["overall"]["value"]
    print(len(dataset))
    s1 = time.perf_counter()
    print("TIME FOR BLEU:", s1 - s0)
    print("BLEU\n", result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate translations (pickle or system output files)"
    )
    parser.add_argument(
        "-f",
        "--format",
        help="file format",
        choices=["pickle", "sys"],
        default="pickle",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="input filename (or dir name for sys outputs), required arg for old format",
    )
    parser.add_argument(
        "-o", "--output", help="output file name", default="translation_results"
    )
    parser.add_argument(
        "--heuristic",
        help="use heuristic (keywords of literal translation) to determine percentage of correct translations",
        action="store_true",
    )
    parser.add_argument(
        "-l", "--lang", help="language used (required to use heuristic)"
    )
    parser.add_argument(
        "--separate_test_sets", action="store_true", help="record results on separate test sets"
    )

    parser.add_argument(
        "-m", "--model", help="model name (required for new test set format)"
    )

    parser.add_argument(
        "--use_critique", action="store_true", help="use critique client rather than hf evaluate"
    )

    args = parser.parse_args()

    if args.heuristic and not args.lang:
        raise ValueError("Need to specify language for heuristic marking")

    auto_eval_fn = automatic_metrics if not args.use_critique else metrics_with_critique

    if args.separate_test_sets: # new format
        translations_df = pd.read_csv(args.input)
        literal_test_set = translations_df[translations_df["type"] == "literal"]
        idiom_test_set = translations_df[translations_df["type"] == "idiomatic"]
        random_test_set = translations_df[translations_df["type"] == "random"]

        print("LITERAL")
        df_lit, *scores_literal = auto_eval_fn(list(literal_test_set["source"]), list(literal_test_set["hypothesis"]), list(literal_test_set["references"]), heuristic=args.heuristic)
        print("IDIOMATIC")
        df_idiom, *scores_idiom = auto_eval_fn(list(idiom_test_set["source"]), list(idiom_test_set["hypothesis"]), list(idiom_test_set["references"]), heuristic=args.heuristic)
        print("RANDOM")
        df_rand, *scores_random = auto_eval_fn(list(random_test_set["source"]), list(random_test_set["hypothesis"]), list(random_test_set["references"]), heuristic=args.heuristic)
    else:
        if args.format == "pickle":
            translations = pickle.load(open(args.input, "rb"))
            hyps = [x["hypothesis"] for x in translations]
            refs = [x["references"][0] for x in translations]
            srcs = [x["source"] for x in translations]
        else:
            hyps = [line.rstrip() for line in open(f"{args.input}/hyps.txt", "r")]
            if "\t" in hyps[0]: # sometimes the score gets left in...
                hyps = [x.split("\t")[1] for x in hyps]
            refs = [line.rstrip() for line in open(f"{args.input}/tgts.txt", "r")]
            srcs = [line.rstrip() for line in open(f"{args.input}/srcs.txt", "r")]
        
        df, *scores = automatic_metrics(srcs, hyps, refs, heuristic=args.heuristic)

    
        df.to_csv(f"./data/opensubtitles_final/{args.output}.csv", index=False)
