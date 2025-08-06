from tiktokenizer.encoder_wrapper import load_gpt2_encoding_offline


tokenizer = load_gpt2_encoding_offline()
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        " of someunkownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))

