import numpy as np


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
            return self._index_to_word[index]

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
