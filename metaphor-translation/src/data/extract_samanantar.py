import pandas as pd
import argparse
import json
from collections import defaultdict
import random
import stanza

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--idioms", type=str, default="./data/external/all_idioms.csv")
parser.add_argument("--output", type=str)
parser.add_argument("--lemma", type=bool, default=True)
parser.add_argument("--random", type=int, help="select N random lines from the corpus (instead of idioms)")
args = parser.parse_args()

def gen_line(filename):
    if not filename:
        return None
    with open(filename) as f:
        for line in f:
            yield line.strip()

def found_idioms(sent, idioms):
    found = False
    idioms_found = []
    sent = sent.replace(",","")
    for idiom in idioms:
        if idiom in sent:
            found = True
            idioms_found.append(idiom)
            unique_idioms[idiom] += 1
    return found, idioms_found

def lemmatize(sent: str) -> str:
    lemma = nlp(sent)
    lemma_str =  " ".join([word.lemma for sent in lemma.sentences for word in sent.words])
    return lemma_str

if __name__ == "__main__":
    idioms = pd.read_csv(args.idioms)
    idioms_hi = [x.lower() for x in idioms.query("lang == 'hi'")["idiom"].tolist()]
    if args.lemma:
        nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos,lemma')
        lemmatized_idioms = [nlp(x) for x in idioms_hi]
        idioms_hi = []
        for l in lemmatized_idioms:
            lemmatized = " ".join([word.lemma for sent in l.sentences for word in sent.words])
            idioms_hi.append(lemmatized)

    total_lines, lines_with_idioms = 0, 0
    unique_idioms = defaultdict(lambda:0)
    random_lines_selected = []

    with open(args.output, "w") as fout:
        i = 0
        for line in gen_line(args.data):
            info = line.strip().split(" ||| ")
            eng_sent, hin_sent = info[0], info[1]
            hin_sent = lemmatize(hin_sent)
            found, idioms_found = found_idioms(hin_sent, idioms_hi)
            output_line = {'src': eng_sent, 'tgt': hin_sent, 'contains_idioms': found, 'idioms': idioms_found}

            if args.random and not found:
                if len(random_lines_selected) < args.random:
                    random_lines_selected.append(output_line)
                else:
                    rand_int = random.randrange(0, i + 1)
                    if rand_int < args.random:
                        random_lines_selected[rand_int] = output_line

            else:
                total_lines += 1
                if found:
                    json.dump(output_line, fout, ensure_ascii=False, separators=(",", ": "))
                    fout.write("\n")
                    lines_with_idioms += 1
                if total_lines % 1000000 == 0:
                    print(f'Processed {total_lines}')
            i += 1

    if not args.random:
        print(f'{lines_with_idioms} out of {total_lines} have at least one idiom!')
        with open(f'unique_idioms_hi.txt', 'w') as fout:
            for idiom, value in unique_idioms.items():
                fout.write(f'{idiom}\t{value}\n')
    else:
        with open(args.output, "w") as fout:
            for output_line in random_lines_selected:
                json.dump(output_line, fout, ensure_ascii=False, separators=(",", ": "))
                fout.write("\n")


