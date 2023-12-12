import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel

import warnings
import numpy
import json
import os

from keras import backend as K

warnings.filterwarnings("ignore")

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

MODEL_NAME = 'roberta-base'
MAX_LEN = 256
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 3
ARTIFACTS_PATH = '../artifacts/'

print('Number of replicas:', strategy.num_replicas_in_sync)

if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)

with open('../dataset/MELD_train_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_train = pd.DataFrame(df, columns=["utterances"])
Y_train = pd.DataFrame(df, columns=["triggers"])
X_train = numpy.array(list(map((lambda x: x[0]), X_train.values)))
Y_train = numpy.array(list(map((lambda x: x[0]), Y_train.values)))

with open('../dataset/MELD_val_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_val = pd.DataFrame(df, columns=["utterances"])
Y_val = pd.DataFrame(df, columns=["triggers"])
X_val = numpy.array(list(map((lambda x: x[0]), X_val.values)))
Y_val = numpy.array(list(map((lambda x: x[0]), Y_val.values)))


def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')  # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)

        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN - 2)])

        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

        # Set to 1s in the attention input
        attention_mask[k, :input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_model(n_categories):
    with strategy.scope():
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

        # Import RoBERTa model from HuggingFace
        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
        x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

        # Huggingface transformers have multiple outputs, embeddings are the first one,
        # so let's slice out the first position
        x = x[0]

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['acc',f1_m,precision_m, recall_m])

        return model


tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

X_train = roberta_encode(X_train, tokenizer)
X_val = roberta_encode(X_val, tokenizer)

y_train = np.asarray(Y_train, dtype='int32')
y_val = np.asarray(Y_val, dtype='int32')

with strategy.scope():
    model = build_model(2)
    model.summary()

with strategy.scope():
    print('Training...')
    history = model.fit(X_train,
                        y_train,
                        # epochs=EPOCHS,
                        epochs=1,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_data=(X_val, y_val))

loss, accuracy, f1_score, precision, recall = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (loss * 100))
print("Accuracy: %.2f%%" % (accuracy * 100))
print("Accuracy: %.2f%%" % (f1_score * 100))
print("Accuracy: %.2f%%" % (precision * 100))
print("Accuracy: %.2f%%" % (recall * 100))
