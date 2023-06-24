# -*- coding: utf-8 -*-
from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse
import stanza
import re
import matplotlib.font_manager
fpaths = matplotlib.font_manager.findSystemFonts()

for i in fpaths:
    f = matplotlib.font_manager.get_font(i)
    print(f.family_name)
    
# from https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

if __name__ == "__main__":
    # Code adapted from the example script in awesome-align repo.
    parser = argparse.ArgumentParser(description="Visualize dot product from awesome-align")
    parser.add_argument("-l", "--lang", required=True, choices=["fr", "jp", "fi"])
    args = parser.parse_args()

    # load model
    model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
    tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

    # model parameters
    align_layer = 8
    threshold = 1e-3

    # default font doesn't display japanese characters
    if args.lang == "jp":
        fprop = matplotlib.font_manager.FontProperties(fname="NotoSansJP-Regular.otf")

    # define inputs
    idiom_df = pd.read_csv(f"./data/opensubtitles_final/{args.lang}_reduced.csv")
    idioms = [x.lower() for x in pd.read_csv("./data/external/all_idioms.csv").query(f"lang == '{args.lang}'")["idiom"].tolist()]
    lang2 = "ja" if args.lang == "jp" else args.lang
    nlp = stanza.Pipeline(lang=lang2, processors='tokenize,pos,lemma', pos_batch_size=1000)

    lemmatized_idioms = [nlp(x) for x in idioms]
    idioms = set()
    for i, l in enumerate(lemmatized_idioms):
        sep = "" if args.lang == "jp" else " "
        lemma_idiom = sep.join([word.lemma for sent in l.sentences for word in sent.words])
        idioms.add(lemma_idiom)

    for ind, row in enumerate(idiom_df.to_dict(orient="records")):
        print(ind)
        if args.lang == "jp":
            src = "".join([c for c in row["original_text"]])
        else:
            src = row["original_text"].lower()
        tgt = row["text"].lower()

        # pre-processing
        src_doc = nlp(src)
        lemma_src_lst = [word.lemma for sent in src_doc.sentences for word in sent.words]
        if args.lang == "jp":
            lemma_src = "".join(lemma_src_lst)
        else:
            lemma_src = " ".join(lemma_src_lst)
        idioms_found = []
        for i in idioms:
            if i in lemma_src:
                idioms_found.append(i)
        if args.lang == "jp":
            sent_src, sent_tgt = [c for c in row["original_text"]], re.findall(r"[\w']+|[.,!?;]", tgt)
        else:
            sent_src, sent_tgt = re.findall(r"[\w']+|[.,!?;]", src), re.findall(r"[\w']+|[.,!?;]", tgt)
        idiom_inds = set()
        for idiom in idioms_found:
            if args.lang != "jp":
                src_idiom_ind = find_sub_list(idiom.split(), lemma_src_lst)
            else:
                pdb.set_trace()
                idiom_chars, lemma_src_chars = list(idiom), list("".join(lemma_src_lst))
                src_idiom_ind = find_sub_list(idiom_chars, lemma_src_chars)
                
            if len(src_idiom_ind) > 0:
                idiom_inds.update(range(src_idiom_ind[0][0], src_idiom_ind[0][1] + 1))

        token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
        token_src_flat, token_tgt_flat = [t for tokenized_word in token_src for t in tokenized_word], [t for tokenized_word in token_tgt for t in tokenized_word]
        wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
        sub2word_map_src = []
        subwords_src = []
        subwords_tgt = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
            sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]
        # alignment
        model.eval()
        with torch.no_grad():
            out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_mult = softmax_srctgt * softmax_tgtsrc
            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)

        align_words = set()
        for i, j in align_subwords:
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
        
        fig, ax = plt.subplots()
        ax.clear()
        ax.matshow(dot_prod)

        if max(dot_prod.shape) <= 10:
            for (i, j), z in np.ndenumerate(dot_prod):
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.set_xticks(np.arange(len(token_tgt_flat)))
        ax.set_xticklabels(token_tgt_flat, rotation=45)

        if args.lang == "jp":
            ax.set_yticks(np.arange(len(token_src_flat)))
            ax.set_yticklabels(token_src_flat, fontproperties=fprop)
        else:
            ax.set_yticks(np.arange(len(token_src_flat)))
            ax.set_yticklabels(token_src_flat)
        pdb.set_trace()
        for ytick, yind in zip(ax.yaxis.get_major_ticks(), sub2word_map_src):
            if yind in idiom_inds:
                ytick.label1.set_color("r")
        plt.tight_layout()
        plt.savefig(f"idiom_align_{args.lang}/idiom_{ind}.png")

        # printing
        class color:
            PURPLE = '\033[95m'
            CYAN = '\033[96m'
            DARKCYAN = '\033[36m'
            BLUE = '\033[94m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'
            END = '\033[0m'

        for i, j in sorted(align_words):
            print(f'{color.BOLD}{color.BLUE}{sent_src[i]}{color.END}==={color.BOLD}{color.RED}{sent_tgt[j]}{color.END}')

    
        