
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
import tensorflow as tf
import numpy as np



tokenizer_path  = "noomo"

##########################################################################################

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(), min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"
    ])

tokenizer.train(["data/synthetic-data.txt"], trainer)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    pad_token = "<pad>",
    mask_token = "<mask>"
)

fast_tokenizer.save_pretrained(tokenizer_path)

##########################################################################################

tokenizer_gpt = GPT2TokenizerFast.from_pretrained(tokenizer_path)

print(f"model.config: vocab.sz={tokenizer_gpt.vocab_size},",
    f"pad_token_id={tokenizer_gpt.pad_token_id},",
    f"bos_token_id={tokenizer_gpt.bos_token_id},",
    f"eos_token_id={tokenizer_gpt.eos_token_id}",
    )

##########################################################################################

def print_tokenization(prompt: str):
    input_ids = tokenizer_gpt(prompt, add_special_tokens=False, padding=False, return_tensors="np")
    input_ids = input_ids["input_ids"]
    input_ids = input_ids[0]

    #print(input_ids)
    print(tokenizer_gpt.convert_ids_to_tokens(input_ids))
    print(tokenizer_gpt.decode(input_ids, skip_special_tokens=False))

print_tokenization("Learning learns hears is hearing")

print_tokenization("Do doing does")
