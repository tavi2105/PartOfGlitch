from keras.models import Sequential
from keras import layers
import json
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy

with open('dataset/MELD_train_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_train = pd.DataFrame(df, columns=["utterances"])
Y_train = pd.DataFrame(df, columns=["triggers"])
X_train = list(map((lambda x: x[0]), X_train.values))
Y_train = numpy.array(list(map((lambda x: x[0]), Y_train.values)))

with open('dataset/MELD_val_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_val = pd.DataFrame(df, columns=["utterances"])
Y_val = pd.DataFrame(df, columns=["triggers"])
X_val = list(map((lambda x: x[0]), X_val.values))
Y_val = numpy.array(list(map((lambda x: x[0]), Y_val.values)))

# print(X_train[2])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

# print(X_train[2])

maxlen = 100
embedding_dim = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# print(type(X_train))
# print(type(Y_train))
history = model.fit(X_train, Y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_val, Y_val),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_val, Y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
