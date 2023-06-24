import pandas as pd
from baselines import init_google
import pdb
import stanza

if __name__ == "__main__":
    idioms = pd.read_csv("./data/external/all_idioms.csv")
    translator = init_google()
    nlp_fr = stanza.Pipeline(
        lang="fr", processors="tokenize,pos,lemma", pos_batch_size=1000
    )
    nlp_fi = stanza.Pipeline(
        lang="fi", processors="tokenize,pos,lemma", pos_batch_size=1000
    )
    nlp_jp = stanza.Pipeline(
        lang="ja", processors="tokenize,pos,lemma", pos_batch_size=1000
    )
    nlp_hi = stanza.Pipeline(
        lang="hi", processors="tokenize,pos,lemma", pos_batch_size=1000
    )
    pipelines = {"fr": nlp_fr, "fi": nlp_fi, "jp": nlp_jp, "hi": nlp_hi}
    content_pos = ["NOUN"]  # just nouns for now...
    salient_word_translations = []

    for i, sample in enumerate(idioms.to_dict(orient="records")):
        lang = sample["lang"]
        if lang != "fi":
            continue
        lang2 = "ja" if lang == "jp" else lang

        nlp = pipelines[lang]
        doc = nlp(sample["idiom"])
        salient_words = []
        for s in doc.sentences:
            print([w.upos for w in s.words])
            for w in s.words:
                if w.upos in content_pos:
                    salient_words.append(w.lemma)
        try:
            word_for_word_trans = [
                translator(w, source_language=lang2) for w in salient_words
            ]
        except:
            print(f"stopped at {i}")
            break
        print(sample["idiom"], word_for_word_trans)
        salient_word_translations.append(word_for_word_trans)
    idioms = idioms.loc[idioms["lang"] == "fi"]
    idioms["noun_translation"] = salient_word_translations
    idioms.to_csv("./data/external/all_idioms_naive_translation_fi.csv", index=False)
