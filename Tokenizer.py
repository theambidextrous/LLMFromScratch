import re


class TokenizerV2:
    def __init__(self, vocab):
        self.string_to_int = vocab
        self.int_to_string = { i:s for s, i in vocab.items() }
    
    def encode(self, text):
        preprosessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprosessed = [
            word.strip() for word in preprosessed if word.strip()
        ]
        # Add unk token for unknown words/tokens
        preprosessed = [ item if item in self.string_to_int else "<|unk|>" for item in preprosessed]
        ids = [self.string_to_int[s] for s in preprosessed ]
        return ids
    
    def decode(self, ids):
        text = " ".join([ self.int_to_string[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class Tokenizer:
    def __init__(self, vocab):
        self.string_to_int = vocab
        self.int_to_string = { i:s for s, i in vocab.items() }
    
    def encode(self, text):
        preprosessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprosessed = [
            word.strip() for word in preprosessed if word.strip()
        ]
        ids = [self.string_to_int[s] for s in preprosessed ]
        return ids
    
    def decode(self, ids):
        text = " ".join([ self.int_to_string[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text