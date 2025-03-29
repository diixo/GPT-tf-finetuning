
# https://www.kaggle.com/code/vimalpillai/finetuning-gpt2-model-tensorflow/notebook

import re
from transformers import GPT2TokenizerFast
import numpy as np


def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []

seq_length = 16

content = [
    "This is a sentence viiX.",
    "The viiX is about of an IT",
    "Here is the another sentence, that is a bit longer.",
    "An energy of IT", "A viiX"]


tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer-gpt")


tokenizer_gpt.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>"
})

encodings = tokenizer_gpt(
    content,
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    max_length=seq_length,
    return_tensors="np"
    )


train_data = encodings["input_ids"][:, :-1]
labels = encodings["input_ids"][:, 1:]
attention_masks = encodings["attention_mask"][:, :-1]

# print(tokenizer_gpt.pad_token_id)
# print(tokenizer_gpt.eos_token_id)

print(encodings["input_ids"])
print(encodings["attention_mask"])

##########################################################################################
mask_tokens_ids = tokenizer_gpt.convert_tokens_to_ids(['a', 'Ġa', 'an', 'Ġan', 'the', 'Ġthe'])

mask_tokens = tokenizer_gpt.convert_ids_to_tokens(mask_tokens_ids)

print(mask_tokens_ids)

# for token_id in mask_tokens_ids:
#     attention_masks[train_data == token_id] = 0

##########################################################################################

def clean_mask_tokens(encodings, mask_tokens, pad_token_id):

    train_data = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]

    batch_size, seq_len = train_data.shape
    new_train_data = np.full((batch_size, seq_len), pad_token_id, dtype=np.int32)
    new_attention_mask = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i in range(batch_size):
        j = 0
        for k in range(seq_len):
            if train_data[i, k] in mask_tokens:
                continue
            if j < seq_len:
                new_train_data[i, j] = train_data[i, k]
                new_attention_mask[i, j] = attention_masks[i, k]
                j += 1
    
    encodings["input_ids"] = new_train_data
    encodings["attention_mask"] = new_attention_mask


clean_mask_tokens(encodings, set(mask_tokens_ids), tokenizer_gpt.pad_token_id)


print("*" * 64)
print(encodings["input_ids"])
print(encodings["attention_mask"])
