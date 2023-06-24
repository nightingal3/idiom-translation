from bs4 import BeautifulSoup
import pandas as pd
import requests
import pdb
import re

idiom_file = "./data/external/idiom_sources.csv"
blocks_scraping_inds = [1, 3, 4, 7, 9, 10, 11, 12, 13, 14, 15]
# open up idiom_file and read into a dataframe
def read_idiom_file():
    df = pd.read_csv(idiom_file)
    return df

def source1(soup):
    h3s =  soup.find_all('h3')
    ps = soup.find_all('p')
    idioms = [idiom.get_text().split(".")[1].strip() for idiom in h3s]
    meanings = [p.get_text().split("Figurative translation:")[1].strip() for p in ps if "Figurative translation:" in p.get_text()]
    misspelled = [p.get_text().split("Figurative translations:")[1].strip() for p in ps if "Figurative translations:" in p.get_text()]
    meanings.extend(misspelled)
    return idioms, meanings, "fr"

def source3(soup):
    bs =  soup.find_all('b')
    ps = soup.find_all(text=True)
    idioms = [idiom.get_text().strip() for idiom in bs[1:]]
    meanings = []
    for b in bs[1:]:
        for item in b.next_siblings:
            if not "–" in item.get_text():
                continue
            else:
                meanings.append(item.split("–")[1].strip())
                break
    return idioms, meanings, "fr"
  
def source6(soup):
    h3s =  soup.find_all('h3')
    ps = soup.find_all('p')
    re_match_num = re.compile(r'\d{1,10}\.?')

    idioms = []
    for h3 in h3s:
        if len(re.findall(re_match_num, h3.get_text())) > 0:
            kanji = h3.get_text().split(".")[1].strip().split("(")[0].strip()
            idioms.append(kanji)
    meanings = ["TODO"] * len(idioms) # this source has meanings that need to be reformatted/cut.

    return idioms, meanings, "jp"

def source7(soup):
    bolded = soup.find_all('strong')
    idioms = [b.get_text().strip() for b in bolded[:30]]
    meanings = []
    for b in bolded:
        try:
            meaning = b.next_sibling.get_text().split("–")[1].strip()
            meanings.append(meaning)
        except:
            continue
    return idioms, meanings, "hi"

def source9(soup):
    bs = soup.find_all('b')
    idioms = []
    meanings = []
    for b in bs[12:]:
        try:
            idiom, meaning = b.get_text().split(" - ")
            idiom = idiom.strip()
            meaning = meaning.strip()
            idioms.append(idiom)
            meanings.append(meaning)
        except:
            continue

    return idioms, meanings, "hi"

def source10(soup):
    trs = soup.find_all('tr')
    idioms = []
    meanings = []
    i = 0
    for t in trs:
        data = t.find_all("td")
        if len(data) > 0: 
            for d in data:
                i += 1
                if i % 3 == 0:
                    meanings.append(d.get_text().strip())
                elif i % 3 == 1:
                    idioms.append(d.get_text().strip())

    return idioms, meanings, "fi"

if __name__ == "__main__":
    df = read_idiom_file()
    parse_fns = [source1, None, source3, None, None, source6, source7, None, source9, None, None, None, None, None, None, None, None, source10]
    idioms = []
    langs = []
    meanings = []

    # For each url, open up the page and scrape the idioms
    for index, row in df.iterrows():
        #pdb.set_trace()
        if index < 17:
            continue
        url, lang = row['url'], row['lang']
        print(url)
        if index in blocks_scraping_inds: # some sites block scraping...or are not worth scraping with a script
            continue 
        parse_fn = parse_fns[index]
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        curr_idioms, curr_meanings, curr_lang = parse_fn(soup)
        idioms.extend(curr_idioms)
        meanings.extend(curr_meanings)
        langs.extend([curr_lang] * len(curr_idioms))
    df = pd.DataFrame({'idiom': idioms, 'lang': langs, 'meaning': meanings})
    df.to_csv("./data/external/idioms_fi.csv", index=False)

