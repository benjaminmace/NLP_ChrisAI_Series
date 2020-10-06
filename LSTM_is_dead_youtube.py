from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers.data.processors import glue_convert_examples_to_features
import tensorflow_datasets
import tensorflow as tf
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

data = tensorflow_datasets.load('glue/mrpc')

train_dataset = glue_convert_examples_to_features(data['train'],
                                                  tokenizer=tokenizer,
                                                  max_length=128,
                                                  task='mrpc')

valid_dataset = glue_convert_examples_to_features(data['validation'],
                                                  tokenizer=tokenizer,
                                                  max_length=128,
                                                  task='mrpc')

train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])

model.fit(train_dataset, epochs=2, validation_data=valid_dataset, callbacks=[ES])

tokenizer_json = tokenizer.to_json()
with open('glue_mrpc.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))