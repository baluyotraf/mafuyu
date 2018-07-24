import numpy as np
from .utils import print_distribution
import os
import enum


class SpecialCharacters(enum.Enum):
    PAD = '<PAD>'
    UNKNOWN = '<UNK>'


class GloveEmbedding:

    def __init__(self, path, encoding='utf-8'):
        index_to_word = [
            SpecialCharacters.PAD,
            SpecialCharacters.UNKNOWN,
        ]
        weights = []
        emb_size = None
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                tokens = line.strip().split(' ')
                word = tokens[0]
                vector = [float(w) for w in tokens[1:]]

                index_to_word.append(word)
                weights.append(vector)

                if emb_size:
                    assert emb_size == len(vector)
                else:
                    emb_size = len(vector)

        self._index_to_word = index_to_word
        self._word_to_index = dict(zip(index_to_word, range(len(index_to_word))))
        self._weights = np.asarray(weights)

        self._vocab_size = len(index_to_word)
        self._embedding_size = emb_size

    def word_to_index(self, word):
        default = self._word_to_index[SpecialCharacters.UNKNOWN]
        return self._word_to_index.get(word, default)

    def index_to_word(self, index):
        if index < 0:
            raise IndexError
        else:
            try:
                return self._index_to_word[index]
            except IndexError:
                return SpecialCharacters.UNKNOWN

    @property
    def weights(self):
        return self._weights

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def embedding_size(self):
        return self._embedding_size


class CharacterEmbedding:

    def __init__(self):
        self._char_count = 0
        self._char_to_index = {}
        self._index_to_char = []

    def _initialize_from_char_list(self, char_list):
        index_to_char = [
            SpecialCharacters.PAD,
            SpecialCharacters.UNKNOWN,
        ]

        char_set = set(char_list)
        index_to_char.extend(char_set)

        self._index_to_char = index_to_char
        self._char_to_index = dict(zip(index_to_char, range(len(index_to_char))))
        self._char_count = len(index_to_char)

    def _initialize_from_sentences(self, sentences):
        char_list = [c for s in sentences for c in s]
        self._initialize_from_char_list(char_list)

    def char_to_index(self, char):
        default = self._char_to_index[SpecialCharacters.UNKNOWN]
        return self._char_to_index.get(char, default)

    def index_to_char(self, index):
        if index < 0:
            raise IndexError
        else:
            try:
                return self._index_to_char[index]
            except IndexError:
                return SpecialCharacters.UNKNOWN

    @property
    def char_count(self):
        return self._char_count

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
