import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open('vocabulary.txt', 'w', encoding='utf-8') as f:
    for token in tokenizer.vocab.keys():
        f.write(token + '\n')