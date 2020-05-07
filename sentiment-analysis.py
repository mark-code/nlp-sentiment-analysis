import numpy as np 
import tensorflow as tf
from tensorflow.python.ops  import *
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import bert


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))


sentences = []
labels = []
with open("Downloads\semeval_twitter_data.txt", encoding="utf-8") as f:
    lines = f.readlines()
    for i, entry in enumerate(lines):
        data_list = []
        for texts in entry.split("\t"):
            data_list.append(texts)
        sentences.append(data_list[3])
        if data_list[2] == '"positive"':
            labels.append(1)
        elif data_list[2] == '"neutral"':
            labels.append(0.66)
        elif data_list[2] == '"negative"':
            labels.append(0)
        elif data_list[2] == '"objective"':
            labels.append(0.33)

training_labels = labels[:6000]
testing_labels = labels[6000:]


training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)


max_length = 130

tokenized_reviews = [tokenize_reviews(review) for review in sentences]

pad = pad_sequences(tokenized_reviews, maxlen = max_length, padding = 'post')

padded = pad[:6000]

testing_padded = pad[6000:]

vocab_size = 31000

embedding_dim = 16

max_length = 130

indices = 5

def batch_generator(data, batch):
    batch_size = int(len(data)/indices)
    batches = []
    for epoch in range(1,indices):
        batches.append(data[((epoch-1)*batch_size):(epoch*batch_size)])
    return(batches[batch-1])


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.summary()

for i in range(1,indices):  
    model.fit(batch_generator(padded, i), batch_generator(training_labels_final, i), epochs=10, validation_data=(batch_generator(testing_padded, i), batch_generator(testing_labels_final, i)))


