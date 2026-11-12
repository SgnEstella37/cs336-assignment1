import pickle

from cs336_basics.tokenization.bpe import train_bpe


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    # train BPE
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    #take the results
    with open("data/tinystories_tokenizer.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)

    print("Tokenizer training completed!")


if __name__ == "__main__":
    main()