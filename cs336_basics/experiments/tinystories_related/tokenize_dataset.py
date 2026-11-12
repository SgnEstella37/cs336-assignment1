import pickle
import numpy as np

from cs336_basics.tokenization.tokenizer import Tokenizer


def main():
    # load the trained BPE
    with open("data/tinystories_tokenizer.pkl", "rb") as f:
        vocab, merges = pickle.load(f)

    # create Tokenizer
    tokenizer = Tokenizer(
        vocab,
        merges,
        special_tokens=["<|endoftext|>"],
    )

    def tokenize_file(tokenizer, input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f:
            token_ids = tokenizer.encode_iterable(f)

            tokens = np.fromiter(
                token_ids,
                dtype=np.uint16,
            )

        np.save(output_path, tokens)

    tokenize_file(
        tokenizer,
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/tinystories_train.npy",
    )

    tokenize_file(
        tokenizer,
        "data/TinyStoriesV2-GPT4-valid.txt",
        "data/tinystories_valid.npy",
    )


if __name__ == "__main__":
    main()
