import torch
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from atten_mech.SelfAttentionMechanism_V1 import SelfAttentionMechanism_V1
from atten_mech.AttentionMechanism_V2 import AttentionMechanism_V2
from atten_mech.CausalAttention import CausalAttention
from atten_mech.MultiHeadAttention import MultiHeadAttention
from atten_mech.MultiHeadWithCombinedWeightsAttention import MultiHeadWithCombinedWeightsAttention as MultiHeadAttentionCombinedQKV


# SELF-ATTENTION WITH TRAINABLE WEIGHTS

# inputs tensor
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # Journey (x^2)
    [0.57, 0.85, 0.64],  # Starts  (x^3)
    [0.22, 0.58, 0.33],  # With    (x^4)
    [0.77, 0.25, 0.10],  # One     (x^5)
    [0.05, 0.80, 0.55]  # Step     (x^6)
])

inputs_batch = torch.stack([inputs, inputs], dim=0)  # (2,6,3)

x_2 = inputs[1]
dimension_in = inputs.shape[1]
dimension_out = 2

torch.manual_seed(123)
# Random init matrices
w_query = torch.nn.Parameter(torch.rand(dimension_in, dimension_out))
w_key = torch.nn.Parameter(torch.rand(dimension_in, dimension_out))
w_value = torch.nn.Parameter(torch.rand(dimension_in, dimension_out))

#Query vector for input 2
query_2 = torch.matmul(x_2, w_query)  # (1,3) x (3,2) = (1,2)
# print("Query vector for input 2: ", query_2)
# Key and Value vectors for all inputs
keys = torch.matmul(inputs, w_key)  # (6,3) x (3,2) = (6,2)
values = torch.matmul(inputs, w_value)  # (6,3) x (3,2) = (6,2)
# print("Keys: ", keys)
# print("Values: ", values)

keys_2 = keys[1]
# Calculating attention scores between query token and key token
# Done by finding dot product -
attention_scores_2 = query_2 @ keys.T  # (1,2) x (2,6) = (1,6)
# print("Attention scores for input 2: ", attention_scores_2)

dime_k = keys.shape[1]

# Scaled Dot-Product Attention using sqrt of dimension of key vectors
# dime_k ** 0.5 == sqrt(dime_k)
attention_weights_2 = torch.softmax(attention_scores_2 / (dime_k ** 0.5), dim=-1)
# print("Attention weights for input 2: ", attention_weights_2)
# print("Attention weights sum for input 2: ", attention_weights_2.sum())

context_vec_2 = attention_weights_2 @ values  # (1,6) x (6,2) = (1,2)
# print("Context vector for input 2: ", context_vec_2)



torch.manual_seed(123)
self_attn_v1 = SelfAttentionMechanism_V1(dimension_in, dimension_out)
all_context_vectors_v1 = self_attn_v1(inputs)
# print("All context vectors V1: ", all_context_vectors_v1)


torch.manual_seed(123)
self_attn_v2 = AttentionMechanism_V2(dimension_in, dimension_out)
all_context_vectors_v2 = self_attn_v2(inputs)
# print("All context vectors V2: ", all_context_vectors_v2)



torch.manual_seed(789)
context_length = inputs_batch.shape[1]
ca = CausalAttention(dimension_in, dimension_out, context_length, 0.0)
all_context_vectors_causal = ca(inputs_batch)
# print("All context vectors with causal attention: ", all_context_vectors_causal)



torch.manual_seed(123)
batch_size, context_length, dimension_in = inputs_batch.shape
dimension_out = 4
multi_head_attn = MultiHeadAttention(dimension_in, dimension_out, context_length, 0.0, num_heads=2)
all_context_vectors_multi_head = multi_head_attn(inputs_batch)
# print("All context vectors with multi-head attention: ", all_context_vectors_multi_head)


batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

mha_combined_qkv = MultiHeadAttentionCombinedQKV(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

out = mha_combined_qkv(embeddings)
print(out.shape)






