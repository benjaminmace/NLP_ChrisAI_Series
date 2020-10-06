from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='darkgrid')
plt.rcParams['figure.figsize'] = (16, 8)

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = 'How many parameters does BERT-large have?'
answer_text = '''BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total
of 340M parameters! Altogether it is 1.34G, so expect it to take a couple of minutes to download to your
 Google Colab instance.'''

input_ids = tokenizer.encode(question, answer_text)

tokens = tokenizer.convert_ids_to_tokens(input_ids)

sep_index = input_ids.index(tokenizer.sep_token_id)

num_seg_a = sep_index + 1

num_seg_b = len(input_ids) - num_seg_a

segment_ids = [0] * num_seg_a + [1] * num_seg_b

assert len(segment_ids) == len(input_ids)

start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "'+answer+'"')

s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

ax = sns.barplot(x=token_labels, y=s_scores, ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
ax.grid(True)
plt.title('Start Word Scores')
plt.show()