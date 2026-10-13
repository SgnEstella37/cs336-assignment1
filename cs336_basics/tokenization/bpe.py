import regex as re

from collections import defaultdict

from torch import t

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(text: str) -> list[str]:
    return re.findall(PAT, text)


def count_tokens(tokens: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    return counts


# first object:word<->times
def convert_to_bytes(counts: dict[str, int]) -> dict[tuple[bytes, ...], int]:
    out = defaultdict(int)
    for key in counts:
        b = key.encode("utf-8")
        byte_tuple = tuple(bytes([x]) for x in b)
        out[byte_tuple] = counts[key]
    return out


# second object:pair<->times
def count_pairs(counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    out = defaultdict(int)
    for key in counts:
        l = len(key)
        for i in range(l - 1):
            tset = (key[i], key[i + 1])
            out[tset] += counts[key]
    return out


# third object:pair->word belonged
def pairs_to_word(words: list[tuple[bytes, ...]]) -> dict[tuple[bytes, bytes], set[int]]:
    out = defaultdict(set)

    for i, w in enumerate(words):
        l = len(w)
        for j in range(l - 1):
            tset = (w[j], w[j + 1])
            out[tset].add(i)

    return out


def merge_tokens(
    first: dict[tuple[bytes, ...], int],
    second: dict[tuple[bytes, bytes], int],
    third: dict[tuple[bytes, bytes], set[int]],
    words: list[tuple[bytes, ...]],
):
    frequencies = list(first.values())
    best_pair = max(second, key=lambda pair: (second[pair], pair))
    theSet = list(third[best_pair])

    for i in theSet:
        old = words[i]
        times = first[old]
        new = []
        j = 0

        # cut down
        for t in range(len(old) - 1):
            pair = (old[t], old[t + 1])
            second[pair] -= times

            third[pair].discard(i)

        # merge
        while j < len(old):
            if j + 1 < len(old) and (old[j], old[j + 1]) == best_pair:
                new.append(best_pair[0] + best_pair[1])
                j += 2
            else:
                new.append(old[j])
                j += 1

        new = tuple(new)
        words[i] = new
        frequency = first.pop(old)
        first[new] = first.get(new, 0) + frequency

        # add over
        for t in range(len(new) - 1):
            pair = (new[t], new[t + 1])
            second[pair] = second.get(pair, 0) + times

            third.setdefault(pair, set()).add(i)
    # clear up
    for pair in list(second.keys()):
        if second[pair] == 0:
            del second[pair]
    # clear up
    for pair in list(third.keys()):
        if not third[pair]:
            del third[pair]

    return best_pair


# BPE main training function
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    # pre tokenize
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    list_it = []

    for chunk in chunks:
        list_it.extend(pre_tokenize(chunk))

    first = convert_to_bytes(count_tokens(list_it))
    second = count_pairs(first)
    words = list(first.keys())
    third = pairs_to_word(words)

    # initialize vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merge = []

    total_time = vocab_size - 256 - len(special_tokens)
    k = 0
    while k < total_time and second:
        the_pair = merge_tokens(first, second, third, words)
        merge.append(the_pair)
        new_token = the_pair[0] + the_pair[1]
        vocab[len(vocab)] = new_token
        k += 1
    return vocab, merge
