from keras.models import Sequential
from keras import layers
import json
import pandas as pd
from keras.backend import clear_session
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


with open('../dataset/MELD_train_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_train = df['emotions'].values
Y_train = df['triggers'].values

with open('../dataset/MELD_val_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_val = df['emotions'].values
Y_val = df['triggers'].values


vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_val = vectorizer.transform(X_val)

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, Y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_val, Y_val),
                    batch_size=10)


clear_session()

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_val, Y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
