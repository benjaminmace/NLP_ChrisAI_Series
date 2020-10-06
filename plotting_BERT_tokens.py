import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.5)
plt.figure(figsize=(10,5))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

token_lengths = [len(token) for token in tokenizer.vocab.keys()]

#sns.countplot(token_lengths)
#plt.title('Vocab Token Lengths')
#plt.xlabel('Token Length')
#plt.ylabel('# of Tokens')

counter = 0
sub_word_lengths = []
for word in tokenizer.vocab.keys():
    if word[:2] == '##':
        counter += 1
        sub_word_lengths.append(len(word[2:]))

sns.countplot(sub_word_lengths)
plt.title('Subword Token Lengths (Without "##")')
plt.xlabel('Subword Lengths')
plt.ylabel('Number of ## Subwords')

plt.show()