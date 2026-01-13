import torch
from GptDatasetV1 import soft_max_naive
from AttentionMechanism import AttentionMechanism

# SELF-ATTENTION MECHANISM
# inputs tensor
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # Journey (x^2)
    [0.57, 0.85, 0.64],  # Starts  (x^3)
    [0.22, 0.58, 0.33],  # With    (x^4)
    [0.77, 0.25, 0.10],  # One     (x^5)
    [0.05, 0.80, 0.55]  # Step     (x^6)
])

# Calculating attention scores between query token and input token
# Done by finding dot product -
query = inputs[1]
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x_i, query)
print(attention_scores_2)

# exit()

# DOT PRODUCT = multiply two vectors element-wise and then sum the product.
# measure of similarity - how close two vectors are aligned, higher value means higher similarity

# Scores Normalization - normalize attention score to get attention weights that sum up to 1
attention_weights_2_tmp = attention_scores_2/attention_scores_2.sum()
print("Weights: ", attention_weights_2_tmp)
print("Weights sum", attention_weights_2_tmp.sum())

# exit()

# SOFTMAX function
attention_weights_2_naive = soft_max_naive(attention_scores_2)
print("Softmax Weights: ", attention_weights_2_naive)
print("Softmax Weights sum", attention_weights_2_naive.sum())

# exit()

# PYTORCH SOFTMAX
attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
print("Torch Softmax Weights: ", attention_weights_2)
print("Torch Softmax Weights sum", attention_weights_2.sum())

# exit()

# CALCULATING CONTEXT VECTOR - summing weighted inputs vectors based on attention weights
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attention_weights_2[i] * x_i
print("Context vector for input 2", context_vec_2)

# Computing for all input vectors
att_mech = AttentionMechanism(inputs)
# print("All attentions scores: ", att_mech.attention_score_v1())
all_attentions_scores = att_mech.attention_score_advanced()
print("All attentions scores: ", all_attentions_scores)

# Normalize all scores to get weights
all_attentions_weights = att_mech.attention_weights()
print("all_attentions_weights:", all_attentions_weights)

# All context vectors by matrix multiplication
all_context_vectors = att_mech.context_vectors()
print("All context vectors:", all_context_vectors)




