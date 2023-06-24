import argparse
import json
import os
import deepl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import pandas as pd
import math
from functools import reduce

from typing import List, Callable
import torch
import string
from tqdm import tqdm
from eaas import Config, Client
import pickle
import pdb
import logging
import time

from google.cloud import translate_v2 as translate
import six

# Unsafe, just to resolve some certificate issues on my end
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
auth_key_deepl = os.environ.get("DEEPL_KEY")

from src.utils.data_utils import find_idioms, read_json

# Adapted from https://cloud.google.com/translate/docs/basic/translate-text-basic#translate_translate_text-python
def init_google(source_language=None):
    translate_client = translate.Client()

    def translate_text_google(text):
        """Translates text into the target language.
        Target must be an ISO 639-1 language code.
        See https://g.co/cloud/translate/v2/translate-reference#supported_languages
        """
        if isinstance(text, six.binary_type):
            text = text.decode("utf-8")

        # Text can also be a sequence of strings, in which case this method
        # will return a sequence of results for each text.
        if source_language is not None:
            result = translate_client.translate(
                text, target_language="en", source_language=source_language
            )
        else:
            result = translate_client.translate(text, target_language="en")
        return result["translatedText"]

    return translate_text_google


def init_model_hf(langcode: str) -> Callable:
    models = {  # TODO: find best models for each language?
        "fr": "Helsinki-NLP/opus-mt-fr-en",
        "jp": "Helsinki-NLP/opus-mt-ja-en",
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "fi": "Helsinki-NLP/opus-mt-fi-en",
    }
    if torch.cuda.is_available():
        translator = pipeline("translation", model=models[langcode], device=0)
    else:
        translator = pipeline("translation", model=models[langcode])

    def translate_text_hf(text) -> str:  # TODO: check return types of these models
        return translator(text)[0]["translation_text"]

    return translate_text_hf


def translate_text_deepl(text):
    translator = deepl.Translator(auth_key_deepl)
    result = translator.translate_text(text, target_lang="EN-US")
    return result.text


def init_substitution(langcode: str) -> Callable:
    idiom_df = pd.read_csv(f"data/external/all_idioms.csv")
    idiom_df = idiom_df.loc[idiom_df["lang"] == langcode]
    # a lot of JP meanings weren't collected
    idiom_df = idiom_df.dropna(subset=["meaning"])
    idioms_sub = {}
    for row in idiom_df.to_dict(orient="records"):
        idioms_sub[row["idiom"]] = row["meaning"]

    idioms_lst = list(idioms_sub.keys())

    def translate_text_substitution(text):
        text = text.lower()
        idioms = find_idioms([text], idioms_lst)
        if idioms[0] == "":
            return text

        for i in idioms[0].split(";"):
            text = text.replace(i, idioms_sub[i])
        return text

    return translate_text_substitution


def translate_all_sents(
    filename: str,
    language: str = "fr",
    baseline: str = "google",
    context_window: int = 0,
    select_idioms: list = [],
) -> List:
    model_translations = []
    with open(filename, "r") as data_file:
        if ".jsonl" in filename:
            data = {i: json.loads(x) for i, x in enumerate(data_file.readlines())}
        elif ".csv" in filename:
            data_df = pd.read_csv(filename)
            data = {}
            # data = data_df.to_dict(orient="records")
            for i, row in enumerate(data_df.to_dict(orient="records")):
                data[i] = {
                    "original_text": row["original_text"],
                    "text": row["text"],
                    "contains_idioms": row["contains_idioms"],
                }
        else:
            data = json.load(data_file)

        logging.info("Translating all sentences...")
        for sample in tqdm(data.values()):
            sample["contains_idioms"] = sample["contains_idioms"].translate(
                str.maketrans("", "", string.whitespace)
            )
            if select_idioms is not None and (
                len(select_idioms) > 0
                and not any(
                    [i in select_idioms for i in sample["contains_idioms"].split(",")]
                )
            ):
                continue

            try:
                text_to_translate = sample["original_text"]
                reference_translation = sample["text"]
            except:
                text_to_translate = sample["tgt"]
                reference_translation = sample["src"]

            if context_window > 0:
                if baseline == "deepl":
                    separator = "\n"
                elif baseline == "hf":
                    if language == "jp":
                        separator = "。"
                    else:
                        separator = "."
                else:
                    seperator = "."

                prev_context = separator.join(
                    sample["context"]["prev"][-context_window:]
                )
                next_context = separator.join(
                    sample["context"]["next"][:context_window]
                )
                text_to_translate_w_context = (
                    prev_context
                    + separator
                    + text_to_translate
                    + separator
                    + next_context
                    + separator
                )

            if baseline == "google":
                translate_fn = init_google()
            elif baseline == "deepl":
                translate_fn = translate_text_deepl
            elif baseline == "substitution":
                sub_fn = init_substitution(language)
                google_translate_fn = init_google()
            else:
                translate_fn = init_model_hf(language)

            if baseline == "substitution":
                model_translation = reduce(
                    lambda r, f: f(r), (sub_fn, google_translate_fn), text_to_translate
                )
            else:
                model_translation = translate_fn(text_to_translate)

            translation_data = {
                "source": text_to_translate,
                "references": [reference_translation],
                "hypothesis": model_translation,
            }
            if context_window > 0:
                model_translation_w_context = translate_fn(text_to_translate_w_context)
                if baseline == "deepl":
                    model_translation_w_context = model_translation_w_context.split(
                        separator
                    )[context_window]
                else:  # other models don't align the translation
                    pass
                translation_data["hypothesis_w_context"] = model_translation_w_context

            # print(text_to_translate, model_translation)
            # print(text_to_translate, model_translation, model_translation_w_context)

            model_translations.append(translation_data)

    return model_translations

def translate_all_df(
    df: pd.DataFrame,
    language: str = "fr",
    baseline: str = "google",
    context_window: int = 0,
    select_idioms: list = [],
) -> pd.DataFrame:
    model_translations = []
    logging.info("Translating all sentences...")
    if baseline == "google":
        if language != "jp":
            translate_fn = init_google(language)
        else:
            translate_fn = init_google("ja")
    elif baseline == "deepl":
        translate_fn = translate_text_deepl
    elif baseline == "substitution":
        sub_fn = init_substitution(language)
        google_translate_fn = init_google(language)
    else:
        translate_fn = init_model_hf(language)

    for i, row in tqdm(df.iterrows()):
        text_to_translate = row["original_text"]
        reference_translation = row["text"]
        
        if baseline == "substitution":
            model_translation = reduce(
                lambda r, f: f(r), (sub_fn, google_translate_fn), text_to_translate
            )
        else:
            model_translation = translate_fn(text_to_translate)

        translation_data = {
            "source": text_to_translate,
            "references": [reference_translation],
            "hypothesis": model_translation,
            "type": row["type"],
        }

        model_translations.append(translation_data)
    
    return model_translations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baselines on collected sentences with idioms/metaphors."
    )
    parser.add_argument(
        "-l",
        "--lang",
        help="source language to translate",
        required=True,
        choices=["fr", "hi", "jp", "fi"],
    )
    parser.add_argument(
        "-t", "--translation_file", help="file of already-generated translations"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="translation model to use",
        default="google",
        choices=["google", "deepl", "hf", "substitution"],
    )
    parser.add_argument(
        "-r", "--random", help="translate random sentences instead", action="store_true"
    )
    parser.add_argument("-o", "--output", help="output file name")
    parser.add_argument(
        "-c", "--context", help="use N context sentences", default=0, type=int
    )
    parser.add_argument(
        "-s",
        "--select_idioms",
        help="limit the idioms to a certain list",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--testset",
        help="Which test set to evaluate on",
        default="idiom",
        choices=["idiom", "literal", "random"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_set_idiomatic = pd.read_csv(f"./data/test_sets_final/{args.lang}/idiomatic_all.csv")
    test_set_idiomatic["type"] = "idiomatic"
    test_set_literal = pd.read_csv(f"./data/test_sets_final/{args.lang}/literal.csv")
    test_set_literal["type"] = "literal"
    test_set_rand = pd.read_csv(f"./data/test_sets_final/{args.lang}/random_sents_ted.csv")
    test_set_rand["type"] = "random"
    test_set_all = pd.concat([test_set_idiomatic, test_set_literal, test_set_rand])

    if not args.output:
        output_name = (
            f"{args.lang}-random-{args.model}-{args.testset}"
            if args.random
            else f"{args.lang}-{args.model}-{args.testset}"
        )
        if args.select_idioms:
            output_name = output_name + f"-selected"
    else:
        output_name = args.output

    if args.translation_file is not None:
        with open(args.translation_file, "rb") as f:
            translations = pickle.load(f)
    else:
        translations = translate_all_df(
            test_set_all, args.lang, args.model, args.context, args.select_idioms
        )
        translations_df = pd.DataFrame(translations)
        translations_idioms = translations_df[translations_df["type"] == "idiomatic"]

        translations_idioms.to_csv(f"data/opensubtitles_final/{args.model}_translations/{args.lang}_idioms.csv", index=False)
        translations_literal = translations_df[translations_df["type"] == "literal"]
        translations_literal.to_csv(f"data/opensubtitles_final/{args.model}_translations/{args.lang}_literal.csv", index=False)
        translations_random = translations_df[translations_df["type"] == "random"]
        translations_random.to_csv(f"data/opensubtitles_final/{args.model}_translations/{args.lang}_random.csv", index=False)

        translations_df.to_csv(f"translation_comparison_{args.lang}_{args.model}.csv", index=False)
        
    print(f"output to translations_{output_name}.p")
    with open(f"translations_{output_name}.p", "wb") as f:
        pickle.dump(translations, f)