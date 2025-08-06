from TheVerdictVocab import TheVerdictVocab
from Tokenizer import Tokenizer, TokenizerV2

vocab = TheVerdictVocab("the-verdict.txt")
tokenizer = Tokenizer(vocab.vocabulary())
tokenizerV2 = TokenizerV2(vocab.vocabulary())
t1 = "Hello, do you like tea?"
t2 = "In the sunlit terraces of the palace"
text = " <|endoftext|> ".join((t1, t2))
print(text)
print(tokenizerV2.encode(text))
print(tokenizerV2.decode(tokenizerV2.encode(text)))

# Text (from context - the verdict story) to token IDs
# ids = tokenizer.encode(text)
# print(ids)

# Token IDs to text
# text = tokenizer.decode(ids)
# print(text)

# Text (out of context context - the verdict story) to token IDs
# this throws key error since word hello is not in context/vocabulary 
# - creating a need for expanding training data
# ids = tokenizer.encode("Hello, do you like tea?")
# print(ids)

