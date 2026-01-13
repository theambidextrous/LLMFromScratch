import torch


class AttentionMechanism:
    def __init__(self, inputs):
        self.inputs = inputs

    def attention_score_v1(self):
        # dot product by loop
        attention_scores = torch.empty(6, 6)
        for i, x_i in enumerate(self.inputs):
            for j, x_j in enumerate(self.inputs):
                attention_scores[i, j] = torch.dot(x_i, x_j)
        return attention_scores

    def attention_score_advanced(self):
        # matrix multiplication of input tensor with its transposed copy
        return self.inputs @ self.inputs.T

    def attention_weights(self):
        # Normalization using torch softmax function with dimension = -1, all rows add up to 1
        return torch.softmax(self.attention_score_advanced(), dim=-1)

    def context_vectors(self):
        # using matrix multiplication of attention weights & input vectors
        return self.attention_weights() @ self.inputs
