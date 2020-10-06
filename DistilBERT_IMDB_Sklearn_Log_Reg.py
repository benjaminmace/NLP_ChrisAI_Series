import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
                 delimiter='\t',
                 header=None)

batch_1 = df.iloc[:, :2000]

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


tokenized = batch_1[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

input_ids = torch.tensor(np.array(padded), dtype=torch.long)

with torch.no_grad():
    last_hidden_state = model(input_ids)

features = last_hidden_state[0][:, 0, :].numpy()

labels = batch_1[1]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression(max_iter=1024)
lr_clf.fit(train_features, train_labels)

print(lr_clf.score(test_features, test_labels))