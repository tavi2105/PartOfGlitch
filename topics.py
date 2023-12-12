import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import json

with open('dataset/MELD_train_efr.json', 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=('utterances', 'emotions', 'triggers'))

index = 0
for dialog in dataset:
    for (utterance, emotion, trigger_value) in zip(dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [utterance, emotion, trigger_value]
        index += 1

X_train = pd.DataFrame(df, columns=["utterances"]).values.tolist()
# print(X_train)

np.random.seed(400)

nltk.download('wordnet')

stemmer = SnowballStemmer("english")


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for utt in text:
        # print(utt)
        for token in gensim.utils.simple_preprocess(utt[0]):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append([lemmatize_stemming(token)])

    return result


def go():
    processed_docs = preprocess(X_train)

    dictionary = gensim.corpora.Dictionary(processed_docs)

    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=10,
                                           id2word=dictionary,
                                           passes=10,
                                           workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")


if __name__ == '__main__':
    go()
