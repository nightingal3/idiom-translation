from bs4 import BeautifulSoup
import os
import pandas as pd
import pdb
import json

data_path = "data/external/aozora-bunko/htmPages"

if __name__ == "__main__":
    parallel_text = {}
    idioms = pd.read_csv("./data/external/all_idioms.csv")
    idioms_jp = [x.lower() for x in idioms.query("lang == 'jp'")["idiom"].tolist()]

    for filename in os.listdir(data_path): # JP encoding: shift-jis
        with open(f"{data_path}/{filename}", "rb") as html:
            soup = BeautifulSoup(html, from_encoding="shift-jis")
            trs = soup.find_all("tr")
            for tr in trs:
                tds = tr.find_all("td")
                if len(tds) == 0:
                    continue
                id = tds[0].get("id")[1:]
                eng_text = tds[0].get_text().strip()
                jp_text = tds[2].get_text().strip()
                contains_idioms = any(idiom in jp_text for idiom in idioms_jp)
                
                parallel_text[id] = {
                    "text": eng_text,
                    "original_text": jp_text,
                    "contains_idioms": contains_idioms
                }

    with open(f"data/external/aozora_bunko_processed/jp-books.json", "w") as f:
            json.dump(parallel_text, f, ensure_ascii=False, indent=4, separators=(",", ": "))