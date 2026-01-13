import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks


def load_gpt2_encoding_offline() -> tiktoken.Encoding:
    encoder_json_path = "/Users/juma/Gigs/LLM/LLMFromScratch/tiktokenizer/encoder.json"
    vocab_bpe_path = "/Users/juma/Gigs/LLM/LLMFromScratch/tiktokenizer/vocab.bpe"
    special_tokens = {"<|endoftext|>": 50256}

    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(vocab_bpe_path, encoder_json_path)

    encoding = tiktoken.Encoding(
        name="gpt2-local",
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^ \p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens
    )

    return encoding
