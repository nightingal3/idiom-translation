from datasets import load_dataset
import json
import pandas as pd

def get_parallel_books(other_dataset, eng_dataset, idioms_list):
    other_to_en = {}
    for book in other_dataset["train"]:
        if "eng" not in book["contentLanguages"]:
            continue
        else:
            other_to_en[book["title"]] = book["text"]

    translated_books = {}
    for book in eng_dataset["train"]:
        if book["title"] in other_to_en:
            translated_books[book["title"]] = {
                "text": book["text"],
                "original_text": other_to_en[book["title"]],
                "contains_idioms": any(idiom in book["text"] for idiom in idioms_list)
            }
    return translated_books

if __name__ == "__main__":
    dataset_hi = load_dataset("sil-ai/bloom-lm", "hin")
    dataset_jp = load_dataset("sil-ai/bloom-lm", "jpn")
    dataset_fr = load_dataset("sil-ai/bloom-lm", "fra")
    dataset_en = load_dataset("sil-ai/bloom-lm", "eng")

    idioms = pd.read_csv("./data/external/all_idioms.csv")
    idioms_hi = [x.lower() for x in idioms.query("lang == 'hi'")["idiom"].tolist()]
    idioms_jp = [x.lower() for x in idioms.query("lang == 'jp'")["idiom"].tolist()]
    idioms_fr = [x.lower() for x in idioms.query("lang == 'fr'")["idiom"].tolist()]

    # zip 3 lists together

    for lang_name, dataset, idiom_list in zip(["hi", "jp", "fr"], [dataset_hi, dataset_jp, dataset_fr], [idioms_hi, idioms_jp, idioms_fr]):
        translated_books = get_parallel_books(dataset, dataset_en, idiom_list)
        with open(f"data/external/bloom/{lang_name}_translated_books.json", "w") as f:
            json.dump(translated_books, f, ensure_ascii=False, indent=4, separators=(",", ": "))
