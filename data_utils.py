import random
from random import shuffle

import torch

SOS_token = 0
SOS = '_'

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: SOS}
        self.n_words = 1  # Count SOS

    def addSentence(self, sentence):
        for word in list(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs
    pairs = [[s for s in l.split(',')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        source_lang = Lang(lang2)
        target_lang = Lang(lang1)
    else:
        source_lang = Lang(lang1)
        target_lang = Lang(lang2)

    return source_lang, target_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    source_lang, target_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        source_lang.addSentence(pair[0])
        target_lang.addSentence(pair[1])
    print("Counted words:")
    print(source_lang.name, source_lang.n_words)
    print(target_lang.name, target_lang.n_words)
    return source_lang, target_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in list(sentence)]


def batchTensorFromSentences(lang, sentences, is_input):
    max_length = max([len(sentence) for sentence in sentences])
    batch_indexes = []
    for sentence in sentences:
        length = len(sentence)
        indexes = indexesFromSentence(lang, sentence)
        # Pad to max length with SOS in the start
        indexes = [SOS_token] * (max_length - length + 1) + indexes
        batch_indexes.append(indexes)

    return torch.tensor(batch_indexes, dtype=torch.long).transpose(0, 1)


def batchTensorsFromPairs(pairs, source_lang, target_lang):
    inputs = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    batch_source_tensor = batchTensorFromSentences(source_lang, inputs, True)
    batch_target_tensor = batchTensorFromSentences(target_lang, targets, False)
    return (batch_source_tensor, batch_target_tensor)
