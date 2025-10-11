import spacy

nlp = spacy.load("en_core_web_sm")


def spacy_noun_phrases(caption):
    doc = nlp(caption)
    nouns = []
    for nounc in doc.noun_chunks:
        nouns.append(str(nounc).lower())
    return nouns


caption = 

spacy_nouns = spacy_noun_phrases(caption)

print(spacy_nouns)
