from GptDatasetV1 import create_data_loader_v1
from tiktokenizer.encoder_wrapper import load_gpt2_encoding_offline
from TheVerdictVocab import TheVerdictVocab

the_verdict = TheVerdictVocab("the-verdict.txt")
raw_text = the_verdict.get_text()
tokenizer = load_gpt2_encoding_offline()
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[:50]
context_size = 4
# overlapping windows
x = enc_sample[:context_size]  # x is the 1st 4
y = enc_sample[1:context_size + 1]  # y is the 2nd 4 elements starting from 2nd
# print(f"x: {x}")
# print(f"y: {y}")

# next-word prediction task will be as follows:

for i in range(1, context_size + 1):
    context = enc_sample[:i]  # represents input to LLM
    desired = enc_sample[i]  # represents what LLM should predict
    print(context, "------->", desired)

# adding a BPE tiktoken decoder
for i in range(1, context_size + 1):
    context = enc_sample[:i]  # represents input to LLM
    desired = enc_sample[i]  # represents what LLM should predict
    print(tokenizer.decode(context), "------->", tokenizer.decode([desired]))

# Using tensors
dataloader = create_data_loader_v1(the_verdict.get_text(), batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)