import json
import base64
import regex as re
from cs336_basics.tokenization.bpe import pre_tokenize


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: [str]
    token_to_id: dict[bytes, int]

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {value: key for key, value in vocab.items()}

    @classmethod
    # Unsure both files are json. form
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # get vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}

        # get merges
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        merges = [(base64.b64decode(a), base64.b64decode(b)) for a, b in merges_data]

        return cls(vocab, merges, special_tokens)
        # implement encode

    def encode(self, text: str) -> list[int]:
        # deal with text first
        if self.special_tokens:
            tokens = sorted(self.special_tokens, key=len, reverse=True)

            pattern = "(" + "|".join(re.escape(token) for token in tokens) + ")"

            parts = re.split(pattern, text)

        else:
            parts = [text]

        tokens_list = []
        for item in parts:
            if item in self.special_tokens:
                tokens_list.append(self.token_to_id[item.encode("utf-8")])
            else:
                for pre_token in pre_tokenize(item):
                    item_b = pre_token.encode("utf-8")
                    word = [bytes([b]) for b in item_b]

                    for pair in self.merges:
                        i = 0
                        while i < len(word) - 1:
                            cp_pair = (word[i], word[i + 1])
                            if pair == cp_pair:
                                new_one = word[i] + word[i + 1]
                                word[i] = new_one
                                word.pop(i + 1)
                            else:
                                i += 1

                    for token in word:
                        tokens_list.append(self.token_to_id[token])

        return tokens_list

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_list=[]
        for num in ids:
            token_list.append(self.vocab[num])

        all_bytes = b"".join(token_list)
        return all_bytes.decode("utf-8", errors="replace")

