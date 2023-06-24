import json
import os
from pathlib import Path

def write_json(parallel_lines: dict, filename: str) -> None:
    with open(f"{filename}.json", "w") as f:
        json.dump(parallel_lines, f, ensure_ascii=False, indent=4, separators=(",", ": "))

def read_json(filename: str) -> None:
    with open(filename, "r") as data_file:
        if ".jsonl" in filename:
            data = {i: json.loads(x) for i, x in enumerate(data_file.readlines())}
        else:
            data = json.load(data_file)
    return data

def safe_mkdir(out_filename: str):
    dir_name = os.path.dirname(out_filename)
    Path(dir_name).mkdir(parents=True, exist_ok=True)

def find_idioms(data, idioms):
    idioms_found = []
    for sample in data:
        curr_idioms = []
        for i in idioms:
            if i in sample.lower():
                curr_idioms.append(i)
        idioms_found.append(";".join(set(curr_idioms)))
    return idioms_found

def write_parallel(data: dict, out_filename_1: str, out_filename_2: str):
    lang1_lines = []
    lang2_lines = []
    for sample in data:
        try:
            lang1 = sample["original_text"]
            lang2 = sample["text"]
        except KeyError:
            lang1 = sample["tgt"]
            lang2 = sample["src"]

        if lang1 == "" or lang2 == "":
            continue
        lang1_lines.append(lang1)
        lang2_lines.append(lang2)

    write_txt(lang1_lines, out_filename_1)
    write_txt(lang2_lines, out_filename_2)


def write_txt(lines: list, out_filename: str):
    safe_mkdir(out_filename)
    with open(out_filename, "w") as f:
        f.write("\n".join(lines))

