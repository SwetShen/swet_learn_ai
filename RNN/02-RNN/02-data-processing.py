import re
import collections

import torch


def read_txt(file_path="./data/word.txt"):
    with open(file_path, "r") as f:
        lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print(f"未知类型错误:{token}")


def count_corpus(tokens):
    if len(tokens) != 0 and isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                continue
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (tuple, list)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def token_freq(self):
        return self._token_freqs


if __name__ == '__main__':
    lines = read_txt()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    print(vocab.token_to_idx)
