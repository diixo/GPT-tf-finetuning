import json
import re


def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []


with open("data/db-full.txt", "r", encoding="utf-8") as f:
    dict_set = set(line.strip() for line in f if line.strip())

print(len(dict_set))

#######################################################################

with open("data/processed-austen-emma.txt", "r", encoding='utf-8') as f:
    text = f.readlines()

words_set = set()
for line in text:
    words = str_tokenize_words(line.lower())
    words_set.update(words)

print(len(words_set))

#######################################################################

difference = words_set - dict_set
print(len(difference))


with open("data/difference.json", "w", encoding="utf-8") as f:
    json.dump(sorted(difference), f, ensure_ascii=False, indent=4)
