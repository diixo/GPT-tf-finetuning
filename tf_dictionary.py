import json
import re

"""
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

"""
#######################################################################
"""
from transformers import MarianMTModel, MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("./opus-mt-en-ru")
model = MarianMTModel.from_pretrained("./opus-mt-en-ru")

texts = ["This is a test.", "Is it a text of news?", "It is a text of news.", "Where is a text of news?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True)
translated = model.generate(**inputs)

translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

for original, translated in zip(texts, translated_texts):
    print(f"{original} → {translated}")

#print(translated_texts)
"""

"""
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from collections import defaultdict
import tensorflow as tf


from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from collections import defaultdict
import tensorflow as tf

tokenizer = T5Tokenizer.from_pretrained("./t5-base")
model = TFT5ForConditionalGeneration.from_pretrained("./t5-base")
model.trainable = False


def lemmatize_batch(words_batch):
    prompt = "lemmatize: " + " ".join(words_batch)
    inputs = tokenizer(prompt, return_tensors="tf", padding=True)
    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=256
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.strip().lower().split()


def group_words_by_lemma(file_path, batch_size=32):
    lemma_groups = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    total = len(words)
    for i in range(0, total, batch_size):
        batch = words[i:i+batch_size]
        lemmas = lemmatize_batch(batch)

        for word, lemma in zip(batch, lemmas):
            lemma_groups[lemma].append(word)

        print(f"Processed {i + len(batch)}/{total}")

    return dict(lemma_groups)


groups = group_words_by_lemma("test.txt")

for lemma, forms in groups.items():
    print(f"{lemma:15} → {forms}")
"""

import spacy
from collections import defaultdict
import json


nlp = spacy.load("en_core_web_sm")

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

    for word in words:
        lemma = lemmatize_spacy(word)
        lemma_groups[lemma].append(word)

    return dict(lemma_groups)


groups = group_words_by_lemma_spacy("data/db-full.txt")

print(f"groups: {len(groups.items())}")

with open("grouped_words.json", "w", encoding="utf-8") as f:
    json.dump(groups, f, ensure_ascii=False, indent=2)
