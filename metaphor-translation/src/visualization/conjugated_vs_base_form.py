import pandas as pd
import argparse
import stanza
import evaluate
import pdb
import ast
from src.models.evaluate_translations import get_paraphrase_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare base form to conjugated form idioms")
    parser.add_argument("-l", "--lang", help="language", choices=["fr", "fi", "jp"], required=True)
    parser.add_argument("-m", "--model", help="model", choices=["google", "deepl", "deltalm"], required=True)
    args = parser.parse_args()
    lang2 = args.lang if args.lang != "jp" else "ja"
    translations_df = pd.read_csv(f"translation_comparison_{args.lang}_{args.model}.csv")
    idiomatic_translations = translations_df[translations_df["type"] == "idiomatic"]
    idioms_lst = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{args.lang}'")["idiom"].tolist()]

    nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)
    
    is_in_base_form = []
    for i, row in idiomatic_translations.iterrows():
        # check if base form is in translation
        base_phrase = row["source"].lower()
        idiom_in_base_phrase = any([idiom in base_phrase for idiom in idioms_lst])
        is_in_base_form.append(idiom_in_base_phrase)

    idiomatic_translations["is_base_form"] = is_in_base_form
    idiomatic_translations = idiomatic_translations.sort_values(by="is_base_form", ascending=False)
    base_forms = idiomatic_translations[idiomatic_translations["is_base_form"] == True]
    conjug_forms = idiomatic_translations[idiomatic_translations["is_base_form"] == False]
    print("num base form idioms:", idiomatic_translations["is_base_form"].sum())
    print("num non-base form idioms: ", len(idiomatic_translations) - idiomatic_translations["is_base_form"].sum())

    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    hyps_base = base_forms["hypothesis"].tolist()
    refs_base = base_forms["references"].tolist()
    hyps_conj = conjug_forms["hypothesis"].tolist()
    refs_conj = conjug_forms["references"].tolist()
    
    #paraphrase_rate, other_scores = get_paraphrase_rate(df, idiom_df, nlp, args.lang, hyps, refs, bleu=bleu)
    #bleu_score = other_scores[0]["bleu"]

    bleu_score_base = bleu.compute(predictions=hyps_base, references=refs_base)
    bleu_score_conj = bleu.compute(predictions=hyps_conj, references=refs_conj)
    bert_score_base = bertscore.compute(predictions=hyps_base, references=refs_base, lang="en")
    bert_score_base = sum(bert_score_base["f1"]) / len(bert_score_base["f1"])
    rouge_score_base = rouge.compute(predictions=hyps_base, references=refs_base)
    results_meteor_base = meteor.compute(predictions=hyps_base, references=refs_base)

    bert_score_conj = bertscore.compute(predictions=hyps_conj, references=refs_conj, lang="en")
    bert_score_conj = sum(bert_score_conj["f1"]) / len(bert_score_conj["f1"])
    rouge_score_conj = rouge.compute(predictions=hyps_conj, references=refs_conj)
    results_meteor_conj = meteor.compute(predictions=hyps_conj, references=refs_conj)

    print(f"BLEU score base: {bleu_score_base}")
    print(f"BLEU score conjugated: {bleu_score_conj}")
    print(f"BERT score base: {bert_score_base}")
    print(f"BERT score conjugated: {bert_score_conj}")
    print(f"ROUGE score base: {rouge_score_base}")
    print(f"ROUGE score conjugated: {rouge_score_conj}")
    print(f"METEOR score base: {results_meteor_base}")
    print(f"METEOR score conjugated: {results_meteor_conj}")

    #print(f"Paraphrase rate: {paraphrase_rate}")
    idiomatic_translations.to_csv(f"translation_comparison_{args.lang}_{args.model}_baseforms.csv", index=False)