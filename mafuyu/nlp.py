import numpy as np
from .utils import print_distribution
import os


class GloveEmbedding:

    def __init__(self, path):
        word_to_index = {}
        index_to_word = []
        weights = []
        emb_size = None
        with open(path) as f:
            for idx, line in enumerate(f):
                tokens = line.strip().split(' ')
                word = tokens[0]
                vector = [float(w) for w in tokens[1:]]

                word_to_index[word] = idx
                index_to_word.append(word)
                weights.append(vector)

                if emb_size:
                    assert emb_size == len(vector)
                else:
                    emb_size = len(vector)
        index_to_word.append(self.unk)

        self._word_to_index = word_to_index
        self._index_to_word = index_to_word
        self._weights = np.asarray(weights)

        self._unk_id = len(weights)
        self._vocab_size = self._unk_id + 1
        self._embedding_size = emb_size

    def word_to_index(self, word):
        return self._word_to_index.get(word, self.unk_id)

    def index_to_word(self, index):
        if index < 0:
            raise IndexError
        else:
            try:
                return self._index_to_word[index]
            except IndexError:
                return self.unk

    @property
    def weights(self):
        return self._weights

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def unk(self):
        return '<UNK>'


class CharacterEmbedding:

    def __init__(self):
        self._char_count = 0
        self._char_to_index = {}
        self._index_to_char = []
        self._unk_id = 0

    def _initialize_from_char_list(self, char_list):
        char_set = set(char_list)

        self._unk_id = len(char_set)
        self._char_count = self._unk_id + 1

        self._index_to_char = [c for c in char_set]
        self._char_to_index = dict(zip(self._index_to_char, range(self._unk_id)))

    def _initialize_from_sentences(self, sentences):
        char_list = [c for s in sentences for c in s]
        self._initialize_from_char_list(char_list)

    def char_to_index(self, char):
        return self._char_to_index.get(char, self.unk_id)

    def index_to_char(self, index):
        if index < 0:
            raise IndexError
        else:
            try:
                return self._index_to_char[index]
            except IndexError:
                return self.unk

    @property
    def char_count(self):
        return self._char_count

    @property
    def unk(self):
        return '<UNK>'

    @property
    def unk_id(self):
        return self._unk_id

    def save(self, path, encoding='utf-8'):
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(path, 'w', encoding=encoding) as file:
            for c in self._index_to_char:
                file.write(c + '\n')

    @classmethod
    def from_sentences(cls, sentences):
        instance = cls()
        instance._initialize_from_sentences(sentences)
        return instance

    @classmethod
    def from_char_list(cls, char_list):
        instance = cls()
        instance._initialize_from_char_list(char_list)
        return instance

    @classmethod
    def load(cls, path, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as file:
            char_list = [f.strip() for f in file]
        return cls.from_char_list(char_list)


def print_length_distribution(data, rows=3, plot_size=(10, 2)):
    lengths = [len(d) for d in data]
    print_distribution(lengths, 'Lengths', rows, plot_size)
