import pandas as pd
import json
import pdb

xlsx_files = {
    "rus": "./data/external/iarpa_project/mets-rus-econ.xlsx",
    "far": "./data/external/iarpa_project/mets-far-econ.xlsx",
    "spa": "./data/external/iarpa_project/mets-spa-econ.xlsx"
}

if __name__ == "__main__":
    for lang, filename in xlsx_files.items():
        print(lang)
        df = pd.read_excel(filename)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        df = df.dropna(subset="english translation")
        df["json_s"] = df.apply(lambda r: {"original_text": r["sentence"], "text": r["english translation"]}, axis=1)
        with open(f"data/external/iarpa_project/{lang}_metaphors.json", "w") as f:
                json.dump(df["json_s"].tolist(), f, ensure_ascii=False, indent=4, separators=(",", ": "))
