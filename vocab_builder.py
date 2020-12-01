import os
import pickle
from collections import Counter
import nltk
from PIL import Image
# nltk.download()

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
def build_vocab(threshold=5):
    counter = Counter()
    with open("./data/flickr8k/captions.txt") as f:
        l = f.readlines()
        for item in l:
            caption=str(item.split(",")[1])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

    # ommit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

vocab = build_vocab()
with open("./data/vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)
