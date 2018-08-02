import numpy as np
from .utils import print_distribution
import os
import enum
import json
from itertools import filterfalse, chain


def _apply_to_nested(data, func=lambda x: x):
    try:
        iter(data)
        if isinstance(data, str):
            return func(data)
        else:
            return [_apply_to_nested(i, func) for i in data]
    except TypeError:
        return func(data)


def _uniquify(iterable):
    seen = set()
    seen_add = seen.add
    for element in filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element


class SpecialTokens(enum.Enum):
    PAD = '<PAD>'
    UNKNOWN = '<UNK>'


class TokenEmbedding:
    def __init__(self, token_list):
        token_list = _uniquify(token_list)
        index_to_token = list(SpecialTokens)
        index_to_token.extend(token_list)
        count = len(index_to_token)

        self._token_to_index = dict(zip(index_to_token, range(count)))
        self._index_to_token = tuple(index_to_token)
        self._token_set = frozenset(index_to_token)

    def token_to_index(self, tokens):
        default = self._token_to_index[SpecialTokens.UNKNOWN]

        def _token_to_index(token):
            return self._token_to_index.get(token, default)

        return _apply_to_nested(tokens, _token_to_index)

    def index_to_token(self, indices, strict=False):

        def _index_to_token(index):
            if index < 0:
                if strict:
                    raise IndexError
                else:
                    return SpecialTokens.UNKNOWN
            else:
                try:
                    return self._index_to_token[index]
                except IndexError as e:
                    if strict:
                        raise IndexError from e
                    else:
                        return SpecialTokens.UNKNOWN

        return _apply_to_nested(indices, _index_to_token)

    def coverage(self, token_list):
        token_list = set(token_list)
        common = self._token_set & token_list
        return len(common) / len(token_list)

    def __contains__(self, item):
        return item in self._token_set

    def __iter__(self):
        return iter(self._index_to_token)

    def __len__(self):
        return len(self._index_to_token)

    def save(self, path, encoding='utf-8'):
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        token_list = self._index_to_token[len(SpecialTokens):]
        with open(path, 'w', encoding=encoding) as file:
            json.dump(token_list, file)

    @classmethod
    def load(cls, path, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as file:
            token_list = json.load(file)
        return cls(token_list)


class GloveEmbedding(TokenEmbedding):

    def __init__(self, word_list, weights):
        weights = np.asarray(weights)

        if weights.shape[0] != len(word_list):
            raise ValueError

        self._weights = weights
        self._embedding_size = weights.shape[1]
        super().__init__(word_list)

    def save(self, path, encoding='utf-8'):
        word_list = self._index_to_token[len(SpecialTokens):]
        weights = self.weights

        def line_generator():
            for word, weight in zip(word_list, weights):
                str_weights = (str(w) for w in weight)
                line = ' '.join(chain([word], str_weights))
                yield line
                yield '\n'

        with open(path, 'w', encoding=encoding) as f:
            f.writelines(line_generator())

    @classmethod
    def load(cls, path, encoding='utf-8'):
        word_list = []
        weights = []
        emb_size = None
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                tokens = line.strip().split(' ')
                word = tokens[0]
                vector = [float(w) for w in tokens[1:]]

                word_list.append(word)
                weights.append(vector)

                if emb_size:
                    assert emb_size == len(vector)
                else:
                    emb_size = len(vector)

        return cls(word_list, weights)

    @property
    def weights(self):
        return self._weights

    @property
    def embedding_size(self):
        return self._embedding_size


def pad_to_minimum(s, min_length):
    length = len(s)
    if length < min_length:
        s.extend([SpecialTokens.PAD] * (min_length - length))
    return s


def print_length_distribution(data, rows=3, plot_size=(10, 2)):
    lengths = [len(d) for d in data]
    print_distribution(lengths, 'Lengths', rows, plot_size)
