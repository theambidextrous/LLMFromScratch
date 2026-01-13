import torch
from TheVerdictVocab import TheVerdictVocab
from GptDatasetV1 import create_data_loader_v1, soft_max_naive

the_verdict = TheVerdictVocab("the-verdict.txt")
raw_text = the_verdict.get_text()
# tiktoken bpe GPT2 vacab size
vocab_size = 50257
# Embed size - aka dimensions - GPT3 uses 12,888 dimensions
output_dimension = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dimension)

# Data loading - pytorch dataset
max_length = 4
dataloader = create_data_loader_v1(
    raw_text, batch_size=8, max_length=max_length,stride=max_length,shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs: \n", inputs)
print("\n Inputs shape: \n", inputs.shape)
# outputs torch.Size([8, 4])

# Embedding layer - embed token IDs into a 256-dimension vectors
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# outputs torch.Size([8, 4, 256])

# For GPT models Absolute-Embedding Approach
# - create another embedding layer with same dimension as that of token embedding layer
# - Alias - Positional Embedding Layer - info about position of each token in the input sequence
context_length = max_length  # sequence of numbers from 0, 1 to maximum input length - 1
pos_embedding_layer = torch.nn.Embedding(context_length, output_dimension)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# Add positional and token embeddings
# Input embeddings are ready to be consumed by LLMs
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
