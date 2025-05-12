import json
import re
import spacy
from collections import defaultdict


#python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []


with open("data/db-full.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]


def extract_diff():

    dict_set = set()

    for w in words:
        print(w) if w in dict_set else dict_set.add(w)

    print(f"dict words: {len(dict_set)}")

    #######################################################################

    with open("data/austen-emma.txt", "r", encoding='utf-8') as f:
        text = f.readlines()

    words_set = set()
    for line in text:
        words = str_tokenize_words(line.lower())
        words_set.update(words)

    print(f"input words: {len(words_set)}")

    #######################################################################

    difference = words_set - dict_set
    print(f"diff (new words): {len(difference)}")

    with open("data/difference.json", "w", encoding="utf-8") as f:
        json.dump(sorted(difference), f, ensure_ascii=False, indent=4)



def find_common_prefix(str1, str2):
    idx = 0
    while idx < min(len(str1), len(str2)) and str1[idx] == str2[idx]:
        idx += 1
    return str1[:idx]

def lemmatize_spacy(word):
    doc = nlp(word)
    return doc[0].lemma_

def group_words_by_lemma_spacy(file_path):
    lemma_groups = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    for i, word in enumerate(words):
        lemma = lemmatize_spacy(word)
        lemma_groups[lemma].append(word)
        if i % 100 == 0:
            print(f"Processed {i}/{len(words)}")

    return dict(lemma_groups)


if __name__ == '__main__':

    groups = group_words_by_lemma_spacy("data/db-full.txt")

    print(f"groups: {len(groups.items())}")

    with open("grouped_words.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
